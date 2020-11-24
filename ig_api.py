import json
import requests
import numpy as np

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


  def _get_current_price(self):
    market_header = self.default_header
    market_header['Version'] = '3'
    market_header['Authorization'] = self.login_auth
    market_header['IG-ACCOUNT-ID'] = self.login_id
    market_res = requests.get('https://demo-api.ig.com/gateway/deal/markets/CS.D.GBPUSD.TODAY.IP', headers=market_header)
    market_json = market_res.json()
    return market_json


  def _get_account(self):
    account_header = self.default_header
    account_header['Version'] = '1'
    account_header['Authorization'] = self.login_auth
    account_header['IG-ACCOUNT-ID'] = self.login_id
    account_res = requests.get('https://demo-api.ig.com/gateway/deal/accounts', headers=account_header)
    account_json = account_res.json()
    return account_json
  

  def _get_account_val(self):
    return np.sum(float(self._get_account()['accounts'][1]['balance']['balance']) + \
                  float(self._get_account()['accounts'][1]['balance']['profitLoss'])) * 0.1
  

  def _get_positions(self):
    positions_header = self.default_header
    positions_header['Version'] = '2'
    positions_header['Authorization'] = self.login_auth
    positions_header['IG-ACCOUNT-ID'] = self.login_id
    positions_res = requests.get('https://demo-api.ig.com/gateway/deal/positions', headers=positions_header)
    positions_json = positions_res.json()
    return positions_json

  
  def _close_positions(self):
    open_positions = self._get_positions()['positions']
    closing_size = 0
    open_buy = 0
    open_sell = 0
    for n in open_positions:
      closing_size += float(n['position']['size'])
      if n['position']['direction'] == 'BUY':
        open_buy += 1
      else:
        open_sell += 1
    if open_buy > open_sell:
      rm_dir = 'SELL'
    else:
      rm_dir = 'BUY'
    
    close_position_header = self.default_header
    close_position_header['Version'] = '2'
    close_position_header['Authorization'] = self.login_auth
    close_position_header['IG-ACCOUNT-ID'] = self.login_id
    
    close_position_body = {
      "epic": "CS.D.GBPUSD.TODAY.IP",
      "expiry": "DFB",
      "direction": rm_dir,
      "size": str(closing_size),
      "orderType": "MARKET",
      "guaranteedStop": "false",
      "forceOpen": "false",
      "currencyCode": "GBP"
    }
    requests.post('https://demo-api.ig.com/gateway/deal/positions/otc', headers=close_position_header, json=close_position_body)

  
  def _open_position(self, pos_dir):
    self._close_positions()
    open_position_header = self.default_header
    open_position_header['Version'] = '2'
    open_position_header['Authorization'] = self.login_auth
    open_position_header['IG-ACCOUNT-ID'] = self.login_id

    open_position_body = {
      "epic": "CS.D.GBPUSD.TODAY.IP",
      "expiry": "DFB",
      "direction": pos_dir,
      "size": "3",
      "orderType": "MARKET",
      "guaranteedStop": "false",
      "forceOpen": "false",
      "currencyCode": "GBP"
    }
    requests.post('https://demo-api.ig.com/gateway/deal/positions/otc', headers=open_position_header, json=open_position_body)