# Data Set Introduction

## UNSW-NB15

### Features

|No.|Name|Type |Description|
|---|---|---|---|
|1|srcip|nominal|Source IP address|
|2|sport|integer|Source port number|
|3|dstip|nominal|Destination IP address|
|4|dsport|integer|Destination port number|
|5|proto|nominal|Transaction protocol|
|6|state|nominal|"Indicates to the state and its dependent protocol| e.g. ACC| CLO| CON| ECO| ECR| FIN| INT| MAS| PAR| REQ| RST| TST| TXD| URH| URN| and (-) (if not used state)"|
|7|dur|Float|Record total duration|
|8|sbytes|Integer|Source to destination transaction bytes |
|9|dbytes|Integer|Destination to source transaction bytes|
|10|sttl|Integer|Source to destination time to live value |
|11|dttl|Integer|Destination to source time to live value|
|12|sloss|Integer|Source packets retransmitted or dropped |
|13|dloss|Integer|Destination packets retransmitted or dropped|
|14|service|nominal|"http| ftp| smtp| ssh| dns| ftp-data |irc  and (-) if not much used service"|
|15|Sload|Float|Source bits per second|
|16|Dload|Float|Destination bits per second|
|17|Spkts|integer|Source to destination packet count |
|18|Dpkts|integer|Destination to source packet count|
|19|swin|integer|Source TCP window advertisement value|
|20|dwin|integer|Destination TCP window advertisement value|
|21|stcpb|integer|Source TCP base sequence number|
|22|dtcpb|integer|Destination TCP base sequence number|
|23|smeansz|integer|Mean of the ?ow packet size transmitted by the src |
|24|dmeansz|integer|Mean of the ?ow packet size transmitted by the dst |
|25|trans_depth|integer|Represents the pipelined depth into the connection of http request/response transaction|
|26|res_bdy_len|integer|Actual uncompressed content size of the data transferred from the server�s http service.|
|27|Sjit|Float|Source jitter (mSec)|
|28|Djit|Float|Destination jitter (mSec)|
|29|Stime|Timestamp|record start time|
|30|Ltime|Timestamp|record last time|
|31|Sintpkt|Float|Source interpacket arrival time (mSec)|
|32|Dintpkt|Float|Destination interpacket arrival time (mSec)|
|33|tcprtt|Float|"TCP connection setup round-trip time| the sum of �synack� and �ackdat�."|
|34|synack|Float|"TCP connection setup time| the time between the SYN and the SYN_ACK packets."|
|35|ackdat|Float|"TCP connection setup time| the time between the SYN_ACK and the ACK packets."|
|36|is_sm_ips_ports|Binary|"If source (1) and destination (3)IP addresses equal and port numbers (2)(4)  equal then| this variable takes value 1 else 0"|
|37|ct_state_ttl|Integer|No. for each state (6) according to specific range of values for source/destination time to live (10) (11).|
|38|ct_flw_http_mthd|Integer|No. of flows that has methods such as Get and Post in http service.|
|39|is_ftp_login|Binary|If the ftp session is accessed by user and password then 1 else 0. |
|40|ct_ftp_cmd|integer|No of flows that has a command in ftp session.|
|41|ct_srv_src|integer|No. of connections that contain the same service (14) and source address (1) in 100 connections according to the last time (26).|
|42|ct_srv_dst|integer|No. of connections that contain the same service (14) and destination address (3) in 100 connections according to the last time (26).|
|43|ct_dst_ltm|integer|No. of connections of the same destination address (3) in 100 connections according to the last time (26).|
|44|ct_src_ ltm|integer|No. of connections of the same source address (1) in 100 connections according to the last time (26).|
|45|ct_src_dport_ltm|integer|No of connections of the same source address (1) and the destination port (4) in 100 connections according to the last time (26).|
|46|ct_dst_sport_ltm|integer|No of connections of the same destination address (3) and the source port (2) in 100 connections according to the last time (26).|
|47|ct_dst_src_ltm|integer|No of connections of the same source (1) and the destination (3) address in in 100 connections according to the last time (26).|
|48|attack_cat|nominal|"The name of each attack category. In this data set | nine categories e.g. Fuzzers| Analysis| Backdoors| DoS Exploits| Generic| Reconnaissance| Shellcode and Worms"|
|49|Label|binary|0 for normal and 1 for attack records|

### Attacks

|Attack category|Attack subcategory|Number of events|
|---|---|---|
|normal||2218761|
| Fuzzers |FTP|558|
| Fuzzers |HTTP|1497|
| Fuzzers |RIP|3550|
| Fuzzers |SMB|5245|
| Fuzzers |Syslog|1851|
| Fuzzers |PPTP|1583|
| Fuzzers| FTP|248|
| Fuzzers|DCERPC|164|
| Fuzzers|OSPF|993|
| Fuzzers |TFTP|193|
| Fuzzers | DCERPC |455|
| Fuzzers | OSPF|1746|
| Fuzzers |BGP|6163|
| Reconnaissance |Telnet|6|
| Reconnaissance |SNMP|69|
| Reconnaissance | SunRPC Portmapper (TCP) UDP Service|2030|
| Reconnaissance | SunRPC Portmapper (TCP) TCP Service |2026|
| Reconnaissance |SunRPC Portmapper (UDP) UDP Service|2045|
| Reconnaissance |NetBIOS|5|
| Reconnaissance |DNS|35|
| Reconnaissance |HTTP|1867|
| Reconnaissance |SunRPC Portmapper (UDP)|2028|
| Reconnaissance | ICMP|1739|
| Reconnaissance | SCTP|367|
| Reconnaissance |MSSQL|5|
| Reconnaissance |SMTP|6|
| Shellcode |FreeBSD|45|
| Shellcode |HP-UX |12|
| Shellcode |NetBSD |45|
| Shellcode |AIX|11|
| Shellcode |SCO Unix|12|
| Shellcode |Linux|236|
| Shellcode |Decoders|102|
| Shellcode |IRIX|12|
| Shellcode |OpenBSD|22|
| Shellcode | Mac OS X|149|
| Shellcode |BSD|252|
| Shellcode |Windows|173|
| Shellcode |BSDi|94|
| Shellcode |Multiple OS|56|
| Shellcode |Solaris|67|
|Analysis|HTML|616|
|Analysis|Port Scanner|2055|
|Analysis|Spam |6|
|Backdoors| |2329|
|DoS| Ethernet|860|
|DoS| Microsoft Office|292|
|DoS| VNC|11|
|DoS|IRC|1|
|DoS|RDP|1319|
|DoS|TCP|30|
|DoS|VNC|2|
|DoS| FTP|69|
|DoS| LDAP|3|
|DoS| Oracle |6|
|DoS| TCP|1712|
|DoS| TFTP|18|
|DoS|DCERPC|6|
|DoS|XINETD|24|
|DoS| IRC|6|
|DoS| SNMP|5|
|DoS|ISAKMP|14|
|DoS|NTP|2|
|DoS|Telnet|13|
|DoS|CUPS|1|
|DoS|Hypervisor|6|
|DoS|ICMP|1478|
|DoS|SunRPC|3|
|DoS| IMAP|7|
|DoS|Asterisk|1210|
|DoS|Browser|1061|
|DoS|Cisco Skinny|6|
|DoS|SIP|60|
|DoS|SMTP|4|
|DoS|SNMP|29|
|DoS|SSL|20|
|DoS|TFTP|3|
|DoS| SMTP|21|
|DoS|DNS|375|
|DoS|IIS Web Server|23|
|DoS|Miscellaneous|3528|
|DoS|RTSP|27|
|DoS| Common Unix Print System (CUPS)|6|
|DoS| SunRPC|17|
|DoS|IGMP|1267|
|DoS|Microsoft Office|49|
|DoS|HTTP|1357|
|DoS|LDAP|17|
|DoS|NetBIOS/SMB|1262|
|DoS|Oracle|1|
|DoS|Windows Explorer|122|
|Exploits| Evasions|212|
|Exploits| SCCP|16|
|Exploits| SSL |33|
|Exploits| VNC|27|
|Exploits|Backup Appliance|162|
|Exploits|Browser |7925|
|Exploits|Clientside Microsoft Office|130|
|Exploits|Interbase|21|
|Exploits|Miscellaneous Batch|1578|
|Exploits|SOCKS|6|
|Exploits|TCP|3|
|Exploits| Apache|1878|
|Exploits| IMAP|36|
|Exploits| Microsoft IIS|296|
|Exploits| SOCKS|1|
|Exploits|Clientside|657|
|Exploits|Clientside Microsoft Paint |8|
|Exploits|IDS|81|
|Exploits|SSH|2|
|Exploits| ICMP|1226|
|Exploits| IDS|18|
|Exploits|DCERPC |121|
|Exploits|FTP|244|
|Exploits|RADIUS|569|
|Exploits|SSL|6|
|Exploits|WINS |3|
|Exploits| Clientside|4480|
|Exploits| Clientside Microsoft |120|
|Exploits| POP3|35|
|Exploits| SSH|12|
|Exploits| TCP|912|
|Exploits| Unix r Service|38|
|Exploits| WINS|18|
|Exploits|Cisco IOS|579|
|Exploits|Clientside Microsoft Media Player |18|
|Exploits|Dameware|6|
|Exploits|IMAP|6|
|Exploits|LPD |6|
|Exploits|MSSQL |26|
|Exploits|Office Document|417|
|Exploits|RTSP|676|
|Exploits|SCADA|932|
|Exploits|VNC|5|
|Exploits|Webserver|465|
|Exploits| All|6|
|Exploits| LDAP|75|
|Exploits| NNTP|18|
|Exploits| Office Document|2433|
|Exploits| RTSP|12|
|Exploits|IGMP|350|
|Exploits|Oracle|187|
|Exploits|RDesktop|14|
|Exploits|Telnet|122|
|Exploits|Unix 'r' Service|6|
|Exploits| LPD|35|
|Exploits|All |1|
|Exploits|Apache|37|
|Exploits|ICMP |3|
|Exploits|Microsoft IIS|51|
|Exploits|PHP|999|
|Exploits|SMB|2130|
|Exploits|SunRPC|74|
|Exploits|Web Application|4639|
|Exploits| PHP|17|
|Exploits|DNS|181|
|Exploits|Evasions|37|
|Exploits|NNTP|3|
|Exploits|SMTP|772|
|Exploits| RADIUS|8|
|Exploits|Browser FTP|1981|
|Exploits|Miscellaneous|5178|
|Exploits|PPTP|13|
|Exploits|SCCP|3|
|Exploits|SIP|1043|
|Exploits|TFTP|87|
|Generic|All|7|
|Generic|SIP|436|
|Generic| HTTP|1|
|Generic|SMTP|247|
|Generic| IXIA|7395|
|Generic| TFTP|116|
|Generic|IXIA|207243|
|Generic|Superflow|10|
|Generic|HTTP|5|
|Generic|TFTP|21|
|Reconnaissance|DNS|6|
|Reconnaissance|SMTP|1|
|Reconnaissance|HTTP|314|
|Reconnaissance|SNMP|12|
|Reconnaissance|SunRPC Portmapper (UDP) TCP Service|349|
|Reconnaissance|MSSQL|1|
|Reconnaissance|NetBIOS |1|
|Reconnaissance|SCTP|2|
|Reconnaissance|SunRPC|2|
|Reconnaissance|Telnet|1|
|Reconnaissance|ICMP|26|
|Reconnaissance|SunRPC Portmapper (TCP) TCP Service|349|
|Reconnaissance|SunRPC Portmapper (TCP) UDP Service|349|
|Reconnaissance| SunRPC Portmapper (UDP) UDP Service |346|
|Shellcode|FreeBSD|8|
|Shellcode|Linux|39|
|Shellcode|OpenBSD|4|
|Shellcode| SCO Unix |2|
|Shellcode|HP-UX|2|
|Shellcode|Mac OS X|26|
|Shellcode|NetBSD|8|
|Shellcode|BSD|44|
|Shellcode|BSDi|16|
|Shellcode|IRIX|2|
|Shellcode|AIX|2|
|Shellcode|Windows|30|
|Shellcode|Decoders|18|
|Shellcode|Multiple OS|10|
|Shellcode|Solaris|12|
|Worms| |174|
||Total |2540044|