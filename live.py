import json
import requests

class LiveData():
  def __init__(self):
    self.default_header = {
      'Content-Type': 'application/json; charset=UTF-8',
      'Accept': 'application/json; charset=UTF-8',
      'X-IG-API-KEY': 'e7d1b155e0eeee82f74d8446fc92e51f957bb043'
    }
    self._login()

  def _login(self):
    login_header = self.default_header
    login_header['Version'] = '3'
    login_res = requests.post('https://demo-api.ig.com/gateway/deal/session', headers=login_header, json={
      "identifier": "mikeylau",
      "password": "Ug37itfmgY92hix"
    })
    login_json = login_res.json()
    self.login_auth = login_json['oauthToken']['token_type'] + ' ' + login_json['oauthToken']['access_token']
    self.login_id = login_json['accountId']

  def _get_live_data(self):
    market_header = self.default_header
    market_header['Version'] = '3'
    market_header['Authorization'] = self.login_auth
    market_header['IG-ACCOUNT-ID'] = self.login_id
    market_res = requests.get('https://demo-api.ig.com/gateway/deal/markets/CS.D.GBPUSD.TODAY.IP', headers=market_header)
    market_json = market_res.json()
    return market_json