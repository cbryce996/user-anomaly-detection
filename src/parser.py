import re
from collections import Counter

# Log file path
log_file_path = "../assets/data/raw/ssh.log"

# Regex for finding lines
failed_username_regex = r"Invalid user (\S+) from (\S+)"
failed_password_regex = r"Failed password for (\S+) from (\S+) port \d+ ssh\d+"
failed_rdns_regex = r"Address (\S+) maps to (\S+), but this does not map back to the address - POSSIBLE BREAK-IN ATTEMPT!"

# Invalid logins
invalid_login_attempts = {}
prepared_output = []

# Open log file
with open(log_file_path, "r") as log:

    # Read lines
    for line in log:
        
        # Find invalid username
        match = re.search(failed_username_regex, line)
        # If match found
        if match:
            # Gets fields
            username = match.group(1)
            ip = match.group(2)

            if ip in invalid_login_attempts:
                invalid_login_attempts[ip]["usernames"].update([username])
            else:
                invalid_login_attempts[ip] = {"usernames": Counter([username]), "passwords": Counter(), "rdns": Counter()}
                
        # Find invalid password
        match = re.search(failed_password_regex, line)
        if match:
            username = match.group(1)
            ip = match.group(2)
            if ip in invalid_login_attempts:
                invalid_login_attempts[ip]["passwords"].update([username])
            else:
                invalid_login_attempts[ip] = {"usernames": Counter(), "passwords": Counter([username]), "rdns": Counter()}

        # Find failed rdns
        match = re.search(failed_rdns_regex, line)
        if match:
            ip = match.group(1)
            if ip in invalid_login_attempts:
                invalid_login_attempts[ip]["rdns"].update([ip])
            else:
                invalid_login_attempts[ip] = {"usernames": Counter(), "passwords": Counter([]), "rdns": Counter(ip)}
    
for ip in invalid_login_attempts.items():
    entry = list({sum(ip[1]["usernames"].values()), sum(ip[1]["passwords"].values()), sum(ip[1]["rdns"].values())})
    prepared_output.append(entry)

for entry in prepared_output:
    print(entry)