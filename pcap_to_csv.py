"""
pcap_to_csv.py — PCAP → Netflow-style CSV for Parssegny netflowv9b_no_duration features.

Produces columns:
  start_time, src_ip, src_port, dst_ip, dst_port, ip_protocol,
  history, conn_state, tcp_flags,
  packet_nb, packet_nb_orig, packet_nb_resp,
  a_l3_payload_size_orig_total, a_l3_payload_size_resp_total,
  a_l3_payload_size_orig_min, a_l3_payload_size_orig_max,
  duration

Usage:
  python3 pcap_to_csv.py -i capture.pcap -o flows.csv [--label malicious|benign]
"""

import argparse
import csv
import socket
import struct
import sys
from collections import defaultdict

import dpkt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ip_to_str(addr: bytes) -> str:
    if len(addr) == 4:
        return socket.inet_ntoa(addr)
    return socket.inet_ntop(socket.AF_INET6, addr)


def flow_key(src: str, sport: int, dst: str, dport: int, proto: int):
    """Canonical 5-tuple — always (lower_ip, lower_port, higher_ip, higher_port)
    so both directions map to the same key. We also store which side is 'orig'."""
    if (src, sport) <= (dst, dport):
        return (src, sport, dst, dport, proto)
    return (dst, dport, src, sport, proto)


# TCP flag bit positions → Zeek-style letters
_TCP_FLAG_BITS = [
    (0x002, "S"),   # SYN
    (0x001, "F"),   # FIN
    (0x004, "R"),   # RST
    (0x008, "P"),   # PSH
    (0x010, "A"),   # ACK
    (0x020, "U"),   # URG
    (0x040, "E"),   # ECE  (kept in raw but excluded from ML features later)
    (0x080, "C"),   # CWR  (same)
]


def tcp_flags_str(flag_set: set) -> str:
    """Convert a set of flag letters to a single string, canonical order SFRAPUEC."""
    order = "SFRPAUEC"
    return "".join(c for c in order if c in flag_set)


# Zeek conn_state derivation (TCP only — simplified but matches Parssegny usage)
def derive_conn_state(saw_syn: bool, saw_synack: bool, saw_fin_orig: bool,
                      saw_fin_resp: bool, rst_orig: bool, rst_resp: bool,
                      data_orig: bool, data_resp: bool) -> str:
    established = saw_syn and saw_synack
    had_data = data_orig or data_resp

    if rst_orig or rst_resp:
        if established and had_data:
            return "RSTO" if rst_orig else "RSTR"
        if established and not had_data:
            return "RSTO" if rst_orig else "RSTR"
        if saw_syn and not saw_synack:
            return "RSTRH" if rst_resp else "REJ"
        return "RSTR"
    if saw_syn and saw_synack:
        if saw_fin_orig and saw_fin_resp:
            return "SF"       # normal close
        if saw_fin_orig or saw_fin_resp:
            return "S1"       # half-closed
        if data_orig or data_resp:
            return "S1"
        return "S2"
    if saw_syn and not saw_synack:
        return "S0"
    if not saw_syn:
        if data_orig or data_resp:
            return "OTH"
        return "OTH"
    return "OTH"


# History string: simplified — we record D (data seen) direction
def derive_history(data_orig: bool, data_resp: bool,
                   saw_syn: bool, saw_synack: bool) -> str:
    h = ""
    if saw_syn:
        h += "S"
    if saw_synack:
        h += "h"      # lowercase = responder direction in Zeek
    if data_orig:
        h += "D"
    if data_resp:
        h += "d"      # lowercase = responder data
    return h if h else "-"


# ---------------------------------------------------------------------------
# Flow accumulator
# ---------------------------------------------------------------------------

class Flow:
    __slots__ = (
        "start_ts", "end_ts",
        "orig", "resp",                    # (ip, port) of originator
        "proto",
        "pkt_orig", "pkt_resp",
        "bytes_orig", "bytes_resp",
        "sizes_orig",                       # list of L3 payload sizes, orig direction
        "tcp_flags_seen",                   # set of flag letters (union both dirs)
        "syn", "synack", "fin_orig", "fin_resp", "rst_orig", "rst_resp",
        "data_orig", "data_resp",
    )

    def __init__(self, ts, orig, resp, proto):
        self.start_ts = ts
        self.end_ts = ts
        self.orig = orig          # (ip_str, port)
        self.resp = resp
        self.proto = proto
        self.pkt_orig = 0
        self.pkt_resp = 0
        self.bytes_orig = 0
        self.bytes_resp = 0
        self.sizes_orig = []
        self.tcp_flags_seen = set()
        self.syn = self.synack = self.fin_orig = self.fin_resp = False
        self.rst_orig = self.rst_resp = False
        self.data_orig = self.data_resp = False


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

FLOW_TIMEOUT = 300   # seconds — expire idle flows


def extract_flows(pcap_path: str) -> list[dict]:
    flows: dict = {}          # key → Flow
    finished: list[Flow] = []

    with open(pcap_path, "rb") as f:
        try:
            cap = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            cap = dpkt.pcapng.Reader(f)

        last_ts = 0.0

        for ts, raw in cap:
            last_ts = ts

            # --- parse Ethernet frame ---
            try:
                eth = dpkt.ethernet.Ethernet(raw)
            except Exception:
                continue

            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                src_ip = ip_to_str(ip.src)
                dst_ip = ip_to_str(ip.dst)
                proto = ip.p
                l3_payload = len(ip.data)
            elif isinstance(eth.data, dpkt.ip6.IP6):
                ip = eth.data
                src_ip = ip_to_str(ip.src)
                dst_ip = ip_to_str(ip.dst)
                proto = ip.nxt
                l3_payload = len(ip.data)
            else:
                continue

            # --- transport layer ---
            tcp_flags_pkt: set = set()
            sport = dport = 0

            if proto == dpkt.ip.IP_PROTO_TCP and isinstance(ip.data, dpkt.tcp.TCP):
                tcp = ip.data
                sport, dport = tcp.sport, tcp.dport
                for bit, letter in _TCP_FLAG_BITS:
                    if tcp.flags & bit:
                        tcp_flags_pkt.add(letter)
                payload_len = len(tcp.data)
            elif proto == dpkt.ip.IP_PROTO_UDP and isinstance(ip.data, dpkt.udp.UDP):
                udp = ip.data
                sport, dport = udp.sport, udp.dport
                payload_len = len(udp.data)
            else:
                continue   # skip non-TCP/UDP

            # --- flow lookup ---
            key = flow_key(src_ip, sport, dst_ip, dport, proto)

            if key not in flows:
                # Determine originator by SYN sender — the side that sends
                # the initial SYN is the true originator (Zeek convention).
                # For the first packet of a new flow, if it has SYN but not
                # ACK, this packet's sender is the originator. Otherwise fall
                # back to IP/port ordering.
                is_syn = "S" in tcp_flags_pkt and "A" not in tcp_flags_pkt
                if is_syn:
                    orig_ep, resp_ep = (src_ip, sport), (dst_ip, dport)
                elif (src_ip, sport) == (key[0], key[1]):
                    orig_ep, resp_ep = (src_ip, sport), (dst_ip, dport)
                else:
                    orig_ep, resp_ep = (dst_ip, dport), (src_ip, sport)
                flows[key] = Flow(ts, orig_ep, resp_ep, proto)

            fl = flows[key]
            is_orig = (src_ip, sport) == (fl.orig[0], fl.orig[1])

            fl = flows[key]
            fl.end_ts = ts
            fl.tcp_flags_seen |= tcp_flags_pkt

            if is_orig:
                fl.pkt_orig += 1
                fl.bytes_orig += l3_payload
                if payload_len > 0:
                    fl.sizes_orig.append(l3_payload)
                    fl.data_orig = True
                if "S" in tcp_flags_pkt and "A" not in tcp_flags_pkt:
                    fl.syn = True
                if "S" in tcp_flags_pkt and "A" in tcp_flags_pkt:
                    fl.synack = True
                if "F" in tcp_flags_pkt:
                    fl.fin_orig = True
                if "R" in tcp_flags_pkt:
                    fl.rst_orig = True
            else:
                fl.pkt_resp += 1
                fl.bytes_resp += l3_payload
                if payload_len > 0:
                    fl.data_resp = True
                if "S" in tcp_flags_pkt and "A" in tcp_flags_pkt:
                    fl.synack = True
                if "F" in tcp_flags_pkt:
                    fl.fin_resp = True
                if "R" in tcp_flags_pkt:
                    fl.rst_resp = True

    # treat all remaining open flows as finished
    finished.extend(flows.values())

    # --- convert to row dicts ---
    rows = []
    for fl in finished:
        if fl.pkt_orig + fl.pkt_resp == 0:
            continue

        conn_state = derive_conn_state(
            fl.syn, fl.synack, fl.fin_orig, fl.fin_resp,
            fl.rst_orig, fl.rst_resp,
            fl.data_orig, fl.data_resp,
        )
        history = derive_history(fl.data_orig, fl.data_resp, fl.syn, fl.synack)
        duration = fl.end_ts - fl.start_ts

        sizes = fl.sizes_orig if fl.sizes_orig else [0]
        rows.append({
            "start_time":                    fl.start_ts,
            "src_ip":                        fl.orig[0],
            "src_port":                      fl.orig[1],
            "dst_ip":                        fl.resp[0],
            "dst_port":                      fl.resp[1],
            "ip_protocol":                   fl.proto,
            "history":                       history,
            "conn_state":                    conn_state,
            "tcp_flags":                     tcp_flags_str(fl.tcp_flags_seen),
            "packet_nb":                     fl.pkt_orig + fl.pkt_resp,
            "packet_nb_orig":                fl.pkt_orig,
            "packet_nb_resp":                fl.pkt_resp,
            "a_l3_payload_size_orig_total":  fl.bytes_orig,
            "a_l3_payload_size_resp_total":  fl.bytes_resp,
            "a_l3_payload_size_orig_min":    min(sizes),
            "a_l3_payload_size_orig_max":    max(sizes),
            "duration":                      round(duration, 6),
        })

    return rows


FIELDNAMES = [
    "start_time", "src_ip", "src_port", "dst_ip", "dst_port", "ip_protocol",
    "history", "conn_state", "tcp_flags",
    "packet_nb", "packet_nb_orig", "packet_nb_resp",
    "a_l3_payload_size_orig_total", "a_l3_payload_size_resp_total",
    "a_l3_payload_size_orig_min", "a_l3_payload_size_orig_max",
    "duration",
]


def write_csv(rows: list[dict], out_path: str, label: str | None) -> None:
    names = FIELDNAMES + (["label"] if label else [])
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for row in rows:
            if label:
                row["label"] = label
            w.writerow(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PCAP → Netflow CSV (netflowv9b_no_duration schema)")
    ap.add_argument("-i", "--input",  required=True,  help="Input .pcap or .pcapng file")
    ap.add_argument("-o", "--output", required=True,  help="Output CSV path")
    ap.add_argument("--label", choices=["malicious", "benign"], default=None,
                    help="Optional label column value to append")
    args = ap.parse_args()

    print(f"[*] Reading {args.input} …", file=sys.stderr)
    rows = extract_flows(args.input)
    print(f"[*] Extracted {len(rows)} flows", file=sys.stderr)
    write_csv(rows, args.output, args.label)
    print(f"[*] Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()