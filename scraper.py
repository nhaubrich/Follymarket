import requests
import json
import os
import pdb
import time

#1. Fetch list of markets (fast, ~1min)
#2. Fetch price data for each market (slow, ~8h)
#3. Merge the two
#
#The price fetching saves as it goes, and can be safely interrupted and resumed

def fetchMarketListings(fileName="marketList.json",delay=1):
    loop=True
    counter = 0
    saveList = []

    while loop:
        print(counter)
        
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
            
        time.sleep(delay)
    
    with open(fileName,"w") as f:
        json.dump(saveList,f,indent=1)
    return None

def fetchPricesForMarket(clob,delay=1):
    rq=requests.get(f"https://clob.polymarket.com/prices-history?interval=all&fidelity=1&market="+clob,timeout=10)
    
    if rq.status_code != 200:
        print(rq.status_code)
        time.sleep(delay)
        return fetchPricesForMarket(clob)
    else:
        data = rq.json()
        history = data['history']
        if len(history)==0:
            print("No history for {}".format(clob))
            return []
        else:
            return history

if __name__=="__main__":
    marketsListFile = "marketList.json"
    pricesListFile = "priceDict.json"
    marketsAndPricesFile = "marketsWithPrices.json"

    if not os.path.isfile(marketsListFile):
        print("Fetching market list")
        fetchMarketListings(marketsListFile)

    with open(marketsListFile,"r") as f:
        markets=json.load(f)

    if not os.path.isfile(pricesListFile):
        with open(pricesListFile,"w") as f:
            f.write("{}\n")

    with open(pricesListFile,"r") as f:
        priceDict = json.load(f)

    fetches = 0
    for i,market in enumerate(markets):
        clobs = json.loads(market["clobTokenIds"])
        outcomes = json.loads(market["outcomes"])
        outcomePrices = json.loads(market["outcomePrices"])
        result = -1
        trueClob = -1
        
        for outcome,price,clob in zip(outcomes,outcomePrices,clobs):
            if price=="1":
                result=outcome
                trueClob = clob

        prices = [] 
        if result!=-1:
            if trueClob not in priceDict:
                fetches+=1
                print("\033[92mfetching {}\033[0m".format(market['slug']))
                priceDict[trueClob] = fetchPricesForMarket(trueClob)
                time.sleep(1)
                if fetches%60==0:
                    completed=False
                    while not completed:
                        try:

                            with open(pricesListFile,"w") as f:
                                json.dump(priceDict,f,indent=1)
                            completed=True
                        except KeyboardInterrupt:
                            Warning("Ignoring interrupt while dumping price json file")
                            continue
        
            prices = priceDict[trueClob]
        
        market["prices"] = prices
        market["result"] = result
        print("{} {} {} {}".format(i,market["slug"],market["result"],len(market["prices"])))

    with open(pricesListFile,"w") as f:
        json.dump(priceDict,f,indent=1)

    marketsWithPrices = [market for market in markets if len(market["prices"])>0]
    with open(marketsAndPricesFile,"w") as f:
        json.dump(markets,f,indent=1)
