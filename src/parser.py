import re
import csv
from collections import Counter

# Log file path
log_file_path = "../assets/data/raw/ssh.log"
csv_output_path = "../assets/data/prepared/ssh.csv"

# Regex for finding lines
failed_username_regex = r"Invalid user (\S+) from (\S+)"
failed_password_regex = r"Failed password for (\S+) from (\S+) port \d+ ssh\d+"
failed_rdns_regex = r"Address (\S+) maps to (\S+), but this does not map back to the address - POSSIBLE BREAK-IN ATTEMPT!"

# Invalid logins
invalid_login_attempts = {}

# Open log file
with open(log_file_path, "r") as log:

    # Read lines
    for line in log:
        
        # Find invalid username
        match = re.search(failed_username_regex, line)

        # If match found
        if match:
            username = match.group(1)
            ip = match.group(2)

            if ip in invalid_login_attempts:
                invalid_login_attempts[ip]["users"].update([username])
            else:
                invalid_login_attempts[ip] = {"users": Counter([username]), "passwords": Counter(), "rdns": Counter()}
                
        # Find invalid password
        match = re.search(failed_password_regex, line)
        if match:
            username = match.group(1)
            ip = match.group(2)

            if ip in invalid_login_attempts:
                invalid_login_attempts[ip]["passwords"].update([username])
            else:
                invalid_login_attempts[ip] = {"users": Counter(), "passwords": Counter([username]), "rdns": Counter()}

        # Find failed rdns
        match = re.search(failed_rdns_regex, line)
        if match:
            ip = match.group(1)
            hostname = match.group(2)

            if ip in invalid_login_attempts:
                invalid_login_attempts[ip]["rdns"].update([hostname])
            else:
                invalid_login_attempts[ip] = {"users": Counter(), "passwords": Counter(), "rdns": Counter([ip])}

header = ["total_users_attempted", "user_password_fails", "rdns_lookup_fails"]

with open(csv_output_path, "w", encoding="UTF8") as file:
        writer = csv.writer(file)

        writer.writerow(header)

        for ip in invalid_login_attempts.items():
            data = [
                sum(ip[1]["users"].values()),
                sum(ip[1]["passwords"].values()),
                sum(ip[1]["rdns"].values())
            ]

            writer.writerow(data)
