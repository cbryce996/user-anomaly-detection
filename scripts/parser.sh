#!/bin/bash

grep "Invalid user" ../assets/data/raw/ssh.log > ../assets/data/parsed/ssh_invalid_user.log

grep "pam_unix(sshd:auth): authentication failure" ../assets/data/raw/ssh.log > ../assets/data/parsed/ssh_auth_fail.log