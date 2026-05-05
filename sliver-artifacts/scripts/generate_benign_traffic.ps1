while ($true) {
    try {
        # Make non-malicious web requests for ajax google apis
	Invoke-WebRequest "https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js" > $null
        Invoke-WebRequest "https://ajax.googleapis.com/ajax/libs/jqueryui/1.13.2/jquery-ui.min.js" > $null
        Invoke-WebRequest "https://ajax.googleapis.com/ajax/libs/angular.js/1.8.3/angular.min.js" > $null

	# Make non-malicious web requests for jquery
	Invoke-WebRequest "https://code.jquery.com/ui/1.13.2/jquery-ui.min.js" > $null
        Invoke-WebRequest "https://code.jquery.com/jquery-3.7.1.min.js" > $null
	Invoke-WebRequest "https://code.jquery.com/jquery-3.6.0.min.js" > $null
	
	# Query the microsoft edge homepage (has a large selection of ads and https traffic generated)
	Invoke-WebRequest "https://www.msn.com" > $null
    } catch {}
    Start-Sleep -Seconds (Get-Random -Minimum 15 -Maximum 60)
}
