# Machine Learning Model

## Data

### Source

The model uses raw SSH logs as a data source, training data has been extracted from a SSH training server on which abnormal access attempts have taken place.

### Raw Data

Raw SSH log of a connection event:

```
Dec 10 07:07:38 LabSZ sshd[24206]: Invalid user test9 from 52.80.34.196
Dec 10 07:07:38 LabSZ sshd[24206]: input_userauth_request: invalid user test9 [preauth]
Dec 10 07:07:38 LabSZ sshd[24206]: pam_unix(sshd:auth): check pass; user unknown
Dec 10 07:07:38 LabSZ sshd[24206]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=ec2-52-80-34-196.cn-north-1.compute.amazonaws.com.cn 
Dec 10 07:07:45 LabSZ sshd[24206]: Failed password for invalid user test9 from 52.80.34.196 port 36060 ssh2
Dec 10 07:07:45 LabSZ sshd[24206]: Received disconnect from 52.80.34.196: 11: Bye Bye [preauth]
```


### Prepared Data

The raw SSH log is prepared to be used by the machine learning model seperating the features of an interaction between a client and the server. Entries are seperate by IP.

**Feture Vector:**

The feature vector is used to provide information to the machine learning model, these represent values which can influence the classification of the result. Each entry is seperated by a unique identity (IP Address).

| Feature | Description | Indication | Data Type
| ------------- | ------------- | ------------- | ------------- |
| total_users_attempted  | Total number of invalid unique user attempts | Indication of a possible dictionary attack | numerical |
| user_password_fails  | Total number of invalid unique user password attempts | Indication of brute forcing a list of known users | numerical |
| root_password_fails  | Total number of invalid root password attempts | Indication of brute forcing the root pasword | numerical |
| rdns_lookup_fails  | Total number of failed RDNS lookups | Indication of spoofed DNS | numerical |





**Classification**

| Feature | Description | Data Type
| ------------- | ------------- | ------------- |
| suspected_attacker  | Indicates whether the data suggests this entry is a possible attacker | boolean |