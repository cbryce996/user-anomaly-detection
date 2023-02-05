# Machine Learning Model

## Data

### Source

The model uses raw SSH logs as a data source, training data has been extracted from a SSH training server on which abnormal access attempts have taken place.

### Raw Data

Raw SSH log:

```
Dec 10 07:07:38 LabSZ sshd[24206]: Invalid user test9 from 52.80.34.196
Dec 10 07:07:38 LabSZ sshd[24206]: input_userauth_request: invalid user test9 [preauth]
Dec 10 07:07:38 LabSZ sshd[24206]: pam_unix(sshd:auth): check pass; user unknown
Dec 10 07:07:38 LabSZ sshd[24206]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=ec2-52-80-34-196.cn-north-1.compute.amazonaws.com.cn 
Dec 10 07:07:45 LabSZ sshd[24206]: Failed password for invalid user test9 from 52.80.34.196 port 36060 ssh2
Dec 10 07:07:45 LabSZ sshd[24206]: Received disconnect from 52.80.34.196: 11: Bye Bye [preauth]
```


### Prerpared Data

The raw SSH log is prepared to be used by the machine learning model seperating the features of an interaction between a client and the server.

| Feature | Description | Data Type
| ------------- | ------------- | ------------- |
| auth_fails  | Total number of SSH authentication fails by IP  | numerical |
| user_fails  | Total number of user fails by IP  | numerical |
| password_fails  | Total number of password fails by IP  | numerical |
| last_user_fail  | Time since last user fail by IP  | numerical |
| last_auth_fail  | Time since last SSH authentication fail by IP  | numerical |