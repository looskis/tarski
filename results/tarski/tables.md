#### banking77

| config | trunk layers run | branch params | intent | mean acc | mean ECE |
|---|---:|---:|---:|---:|---:|
| probe@4 | 4 | 0.06M | 87.9 (-6.1) | 87.9 | 0.014 |
| probe@8 | 8 | 0.06M | 88.6 (-5.4) | 88.6 | 0.021 |
| probe@11 | 11 | 0.06M | 88.1 (-5.9) | 88.1 | 0.011 |
| probe@14 | 14 | 0.06M | 87.3 (-6.8) | 87.3 | 0.019 |
| probe@18 | 18 | 0.06M | 87.5 (-6.6) | 87.5 | 0.011 |
| probe@22 | 22 | 0.06M | 88.7 (-5.4) | 88.7 | 0.010 |
| blocks@4+1 | 4 | 5.08M | 90.2 (-3.8) | 90.2 | 0.016 |
| blocks@4+2 | 4 | 10.09M | 91.1 (-3.0) | 91.1 | 0.007 |
| blocks@8+1 | 8 | 5.08M | 91.0 (-3.1) | 91.0 | 0.019 |
| blocks@8+2 | 8 | 10.09M | 92.0 (-2.0) | 92.0 | 0.013 |
| blocks@11+1 | 11 | 5.08M | 90.6 (-3.4) | 90.6 | 0.017 |
| blocks@11+2 | 11 | 10.09M | 91.4 (-2.6) | 91.4 | 0.011 |
| blocks@14+1 | 14 | 5.08M | 90.3 (-3.7) | 90.3 | 0.020 |
| blocks@14+2 | 14 | 10.09M | 92.8 (-1.2) | 92.8 | 0.016 |
| blocks@18+1 | 18 | 5.08M | 93.1 (-0.9) | 93.1 | 0.015 |
| blocks@18+2 | 18 | 10.09M | 92.9 (-1.1) | 92.9 | 0.010 |
| blocks@20+1 | 20 | 5.08M | 91.5 (-2.5) | 91.5 | 0.018 |
| blocks@20+2 | 20 | 10.09M | 92.4 (-1.7) | 92.4 | 0.015 |
| full | 0 | 110.39M | 94.0 | 94.0 | 0.009 |

Parenthesised: accuracy points relative to the full fine-tune of that task.

#### clinc150

| config | trunk layers run | branch params | domain | intent | oos | mean acc | mean ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| probe@6 | 6 | 0.12M | 82.6 | 86.2 | 85.8 | 84.9 | 0.058 |
| probe@11 | 11 | 0.12M | 82.0 | 86.1 | 85.1 | 84.4 | 0.061 |
| probe@16 | 16 | 0.12M | 82.9 | 84.8 | 86.3 | 84.7 | 0.060 |
| probe@22 | 22 | 0.12M | 83.8 | 85.7 | 87.0 | 85.5 | 0.049 |
| blocks@6+1 | 6 | 5.13M | 86.8 | 85.5 | 85.7 | 86.0 | 0.059 |
| blocks@6+2 | 6 | 10.15M | 87.9 | 86.3 | 86.1 | 86.8 | 0.057 |
| blocks@11+1 | 11 | 5.13M | 86.8 | 87.0 | 86.2 | 86.7 | 0.054 |
| blocks@11+2 | 11 | 10.15M | 87.6 | 87.1 | 86.2 | 87.0 | 0.057 |

#### typed

| config | trunk layers run | branch params | agent_trace_observability.action | agent_trace_observability.needs_review | agent_trace_observability.outcome | agent_trace_observability.risk | agent_trace_observability.urgency | customer_service.action | customer_service.category | customer_service.churn_risk | customer_service.needs_human | customer_service.urgency | invoice_processing.discrepancy_severity | invoice_processing.disposition | invoice_processing.duplicate | invoice_processing.matches_order | invoice_processing.urgency | security_incidents.credential_compromise | security_incidents.disposition | security_incidents.severity | security_incidents.true_positive | security_incidents.urgency | mean acc | mean ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| probe@6 | 6 | 0.01M | 36.0 | 55.0 | 35.0 | 44.0 | 36.0 | 52.0 | 51.0 | 53.0 | 67.0 | 42.0 | 47.0 | 43.0 | 89.0 | 46.0 | 32.0 | 55.0 | 74.0 | 49.0 | 73.0 | 50.0 | 51.5 | 0.085 |
| probe@11 | 11 | 0.01M | 36.0 | 73.0 | 28.0 | 35.0 | 36.0 | 52.0 | 32.0 | 47.0 | 67.0 | 42.0 | 47.0 | 45.0 | 89.0 | 74.0 | 32.0 | 55.0 | 74.0 | 49.0 | 73.0 | 50.0 | 51.8 | 0.100 |
| probe@16 | 16 | 0.01M | 36.0 | 47.0 | 26.0 | 40.0 | 36.0 | 52.0 | 31.0 | 50.0 | 67.0 | 42.0 | 47.0 | 38.0 | 89.0 | 73.0 | 32.0 | 55.0 | 74.0 | 49.0 | 73.0 | 50.0 | 50.4 | 0.104 |
| probe@22 | 22 | 0.01M | 36.0 | 70.0 | 28.0 | 40.0 | 36.0 | 52.0 | 73.0 | 47.0 | 67.0 | 51.0 | 47.0 | 38.0 | 89.0 | 61.0 | 59.0 | 55.0 | 74.0 | 49.0 | 77.0 | 51.0 | 55.0 | 0.095 |
