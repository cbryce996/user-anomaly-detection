import re
import csv
from collections import defaultdict, Counter

def read_log_file(log_file_path):
    # Regex for finding lines
    failed_username_regex = r"Invalid user (\S+) from (\S+)"
    failed_password_regex = r"Failed password for (\S+) from (\S+) port \d+ ssh\d+"
    failed_rdns_regex = r"Address (\S+) maps to (\S+), but this does not map back to the address - POSSIBLE BREAK-IN ATTEMPT!"

    # Invalid logins
    invalid_login_attempts = defaultdict(lambda: {"users": Counter(), "passwords": Counter(), "rdns": Counter()})

    # Open log file
    with open(log_file_path, "r") as log:
        # Read lines
        for line in log:
            # Find invalid username
            match = re.search(failed_username_regex, line)

            # If match found
            if match:
                username, source_ip_address = match.groups()

                invalid_login_attempts[source_ip_address]["users"].update([username])

            # Find invalid password
            match = re.search(failed_password_regex, line)
            if match:
                username, source_ip_address = match.groups()

                invalid_login_attempts[source_ip_address]["passwords"].update([username])

            # Find failed rdns
            match = re.search(failed_rdns_regex, line)
            if match:
                source_ip_address, hostname = match.groups()

                invalid_login_attempts[source_ip_address]["rdns"].update([hostname])

    return invalid_login_attempts

def write_csv_file(csv_output_path, invalid_login_attempts):
    header = ["total_users_attempted", "user_password_fails", "rdns_lookup_fails"]

    with open(csv_output_path, "w", encoding="UTF8") as file:
        writer = csv.writer(file)

        writer.writerow(header)

        for ip, data in invalid_login_attempts.items():
            writer.writerow([
                sum(data["users"].values()),
                sum(data["passwords"].values()),
                sum(data["rdns"].values())
            ])

# Log file path
log_file_path = "../assets/data/raw/ssh (1).log"
csv_output_path = "../assets/data/prepared/ssh.csv"

# Read log file
invalid_login_attempts = read_log_file(log_file_path)

# Write CSV file
write_csv_file(csv_output_path, invalid_login_attempts)
