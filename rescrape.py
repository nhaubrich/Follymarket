import requests
import json
import os
import pdb
import time

#getting markets is fast
#getting prices per clob is slow and unreliable

#1. fetch markets into markets.json
#2. make dict of market-to-prices
#3. mix together

def getMarkets():
    loop=True
    counter = 0
    saveList = []

    while loop:
        print(counter)
        
        #rq = requests.get("https://gamma-api.polymarket.com/markets?limit=500&closed=True&offset="+str(counter*500))

        min_start_date="2023-12-27"
        print("min start date: {}".format(min_start_date))
        rq = requests.get("https://gamma-api.polymarket.com/markets?limit=500&closed=True&offset="+str(counter*500)+"&start_date_min={}".format(min_start_date))
        if rq.status_code != 200:
            print(rq.status_code)
        else:
            counter+=1
            markets = rq.json()
            saveList+=markets
            if len(markets)==0:
                loop=False
            #for market in markets:
                #clobs = json.loads(market["clobTokenIds"])
            
        time.sleep(1)
    
    with open("marketList.json","w") as f:
        json.dump(saveList,f,indent=1)
    return None

def getPrice(clob):
    rq=requests.get(f"https://clob.polymarket.com/prices-history?interval=all&fidelity=1&market="+clob,timeout=10)
    
    if rq.status_code != 200:
        print(rq.status_code)
        time.sleep(1)
        return getPrice(clob)
    else:
        data = rq.json()
        history = data['history']
        if len(history)==0:
            print("No history for {}".format(clob))
            return []
        else:
            return history

if not os.path.isfile("marketList.json"):
    print("fetching market list")
    getMarkets()

with open("marketList.json","r") as f:
    markets=json.load(f)

if not os.path.isfile("priceDict.json"):
    with open("priceDict.json","w") as f:
        f.write("{}\n")


with open("priceDict.json","r") as f:
    priceDict = json.load(f)

fetches = 0
for i,market in enumerate(markets):

    clobs = json.loads(market["clobTokenIds"])
    outcomes = json.loads(market["outcomes"])
    outcomePrices = json.loads(market["outcomePrices"])
    result = -1
    trueClob = -1
    #hmm, this gives the price of no
    for outcome,price,clob in zip(outcomes,outcomePrices,clobs):
        #endprice = float(price)
        #if round(endprice)==1:
        if price=="1":
            result=outcome
            trueClob = clob
        #if price=="0.5": #ignore ties for now

    prices = [] 
    if result!=-1:
        if trueClob not in priceDict:
            fetches+=1
            print("fetching {}".format(market['slug']))
            priceDict[trueClob] = getPrice(trueClob)
            time.sleep(1)
            if fetches%60==0:
                with open("priceDict.json","w") as f:
                    json.dump(priceDict,f,indent=1)
    
        prices = priceDict[trueClob]

    
    market["prices"] = prices
    market["result"] = result
    print("{} {} {} {}".format(i,market["slug"],market["result"],len(market["prices"])))

with open("priceDict.json","w") as f:
    json.dump(priceDict,f,indent=1)

marketsWithPrices = [market for market in markets if len(market["prices"])>0]
with open("marketsWithPrices.json","w") as f:
    json.dump(markets,f,indent=1)
