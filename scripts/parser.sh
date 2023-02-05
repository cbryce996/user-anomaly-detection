#!/bin/bash

# Gets invalid user fails
grep "Invalid user" ../assets/data/raw/ssh.log > ../assets/data/parsed/ssh_invalid_user.log

# Gets ssh auth fails
grep "pam_unix(sshd:auth): authentication failure" ../assets/data/raw/ssh.log > ../assets/data/parsed/ssh_auth_fail.log

# Gets unique ip's
grep -E -o "([0-9]{1,3}[\.]){3}[0-9]{1,3}" ../assets/data/raw/ssh.log | sort -u > ../assets/data/parsed/ssh_ip.log

# Check if output location exists
if [[ ! -e ../assets/data/prepared/data.csv ]]; then
    mkdir -p ../assets/data/prepared
    touch ../assets/data/prepared/data.csv
fi

# If it does empty file
> ../assets/data/prepared/data.csv

# Echo features to output file
cat ../assets/data/parsed/ssh_ip.log | while read line 
do
   invalid_user_count=$(grep -c "$line" ../assets/data/parsed/ssh_invalid_user.log)
   auth_fail_count=$(grep -c "$line" ../assets/data/parsed/ssh_auth_fail.log)
   echo "$line,$invalid_user_count,$auth_fail_count" >> ../assets/data/prepared/data.csv
done
