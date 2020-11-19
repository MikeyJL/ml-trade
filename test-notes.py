"""
Need to add different amounts to the system
Q1XXYJ8NY07Q1UMS
"""

from sklearn.preprocessing import StandardScaler

max_price = 400
min_price = 240

scaler = StandardScaler()
scaler.fit([[min_price], [max_price]])
print(scaler.transform([[240], [320], [400]]))