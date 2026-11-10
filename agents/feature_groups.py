
# Feature groups for four cybersecurity specialist agents.
# Target columns and ID are deliberately excluded.

TRAFFIC_FEATURES = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "smean",
    "dmean"
]


THREAT_FEATURES = [
    "proto",
    "service",
    "state",
    "sttl",
    "dttl",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "tcprtt",
    "synack",
    "ackdat",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd"
]


BEHAVIOUR_FEATURES = [
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "trans_depth",
    "response_body_len"
]


RISK_FEATURES = [
    "dur",
    "rate",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "ct_state_ttl",
    "ct_dst_src_ltm",
    "is_sm_ips_ports",
    "sload",
    "dload"
]


AGENT_FEATURES = {
    "Traffic Agent": TRAFFIC_FEATURES,
    "Threat Agent": THREAT_FEATURES,
    "Behaviour Agent": BEHAVIOUR_FEATURES,
    "Risk Agent": RISK_FEATURES
}


TARGET_COLUMN = "label"

EXCLUDED_COLUMNS = [
    "id",
    "label",
    "attack_cat"
]
