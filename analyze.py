import json
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
import matplotlib.dates as mdates
import pdb

majors = mdates.DateFormatter('%d')
minors = mdates.DateFormatter('%h')

with open("marketsWithPrices.json","r") as f:
    data = json.load(f)

counter=0
for i,market in enumerate(data):

    if "prices" not in market.keys():
        market["prices"] = []
        market["result"] = "-1"

    elif len(market["prices"])>0:
        counter+=1
print("{} markets with price data".format(counter))

#summary plot: histogram of true percentage vs actual percentage

#calibration plan:
#randomly sample one price/prob from market
#repeat many times
#calibration at p is # p's divide by (# p's + # (1-p)'s)
durations = []
prices = []
pricesPerMarket = 10000
priceList = []
maxMarketLength = 0
volumeList = []

for market in data:
    if len(market["prices"])>0 and "volume" in market:
        listOfPrices = [p["p"] for p in market["prices"]]
        #prices+=random.choices([p["p"] for p in market["prices"]],k=pricesPerMarket)
        prices+=listOfPrices
        priceList.append(listOfPrices[::-1])
        volumeList.append(float(market["volume"]))
        durations.append(len(listOfPrices)*1/6)
        maxMarketLength=max(maxMarketLength,len(listOfPrices))


        #another measure is go by median
        median = sorted(listOfPrices)[len(listOfPrices)//2]
        
        if False: #min(listOfPrices)<0.03 or median<0.1:
            print(" ".join([market["outcomes"],market["result"],str(market["prices"][-1]['p']),market["outcomePrices"],market["slug"]]))
            times = [datetime.datetime.utcfromtimestamp(t['t']) for t in market['prices']]

            plt.plot(times,listOfPrices)
            plt.gcf().autofmt_xdate()
            plt.title(market['slug']+"\n min prob: {:.3f}     median prob: {:.3f}".format(min(listOfPrices),median))
            plt.savefig("plots/folly_{}.png".format(market['slug']))
            #plt.show()
            #pdb.set_trace()
            plt.clf()

def computeAccuracy(prices):
    h,bins=np.histogram(prices,range=(0,1),bins=51)
    midpoints = 0.5*(bins[:-1]+bins[1:])
    ratios = h/(h+h[::-1])
    return ratios,midpoints

def bootstrap(data, draws):
    N = len(data)
    ratios, midpoints = computeAccuracy(data)
    sum_squared_variation = np.zeros(ratios.shape)
   
    block = 100
    for i in range(0,draws,block):
        print(i)
        samples = np.random.choice(data,size=(min(block,draws-i),N))
        for sample in samples:
            ratios_draw,_ = computeAccuracy(sample)
            sum_squared_variation += (ratios_draw-ratios)**2
    
    bin_errors =  (1/(draws-1)*sum_squared_variation)**0.5
    return bin_errors

def quantileBootstrap(data, draws):
    N = len(data)
    ratios, midpoints = computeAccuracy(data)
    bootstraps = np.zeros((draws,len(midpoints)))

    block = 100
    for i in range(0,draws,block):
        print(i)
        samples = np.random.choice(data,size=(min(block,draws-i),N))
        for j,sample in enumerate(samples):
            ratios_draw,_ = computeAccuracy(sample)
            bootstraps[i+j] = ratios_draw
            #sum_squared_variation += (ratios_draw-ratios)**2
    
    lower,upper = np.quantile(bootstraps,[0.16,0.84],axis=0)-ratios
    #pdb.set_trace()
    return np.abs(np.vstack((lower,upper)))

def flatten(listOfLists):
    return [e for l in listOfLists for e in l]

def quantileBootstrapMarkets(priceList, draws):
    N = len(priceList)
    
    ratios, midpoints = computeAccuracy(flatten(priceList))
    bootstraps = np.zeros((draws,len(midpoints)))

    block = 100
    for i in range(0,draws,block):
        print(i)
        #samples = np.random.choice(data,size=(min(block,draws-i),N))

        #choose indices of markets
        sampleIdxs = np.random.randint(0,N,size=(min(block,draws-i),N))
        

        for j,sampleIdx in enumerate(sampleIdxs):
            sample = [priceList[idx] for idx in sampleIdx]
            ratios_draw,_ = computeAccuracy(flatten(sample))
            bootstraps[i+j] = ratios_draw
            #sum_squared_variation += (ratios_draw-ratios)**2
    
    lower,upper = np.quantile(bootstraps,[0.16,0.84],axis=0)-ratios
    #pdb.set_trace()
    return np.abs(np.vstack((lower,upper)))

print("longest market: {}".format(maxMarketLength))

#bootstrap markets, not individual prices, due to correlations
ratios,midpoints = computeAccuracy(prices)
#errors = quantileBootstrap(prices,50)

#errors = quantileBootstrapMarkets(priceList,5)
#plt.errorbar(midpoints,ratios,yerr=errors)

plt.scatter(midpoints,ratios)

plt.plot([0,1],[0,1],linestyle='dashed',color='green')
plt.xlabel("Market Probability")
plt.ylabel("Outcome Frequency")
plt.savefig("plots/calibration.png")
#plt.show()
plt.clf()

plt.hist(durations,histtype='step',bins=100)
plt.xlabel("Market duration (h)")
plt.savefig("plots/duration.png")
#plt.show()
plt.clf()


#predictit investigation https://researchers.one/articles/18.11.00005.pdf
#brier score (MSE), do it daily? https://en.wikipedia.org/wiki/Brier_score
#brier score with many markets and varying timespans https://dspace.mit.edu/bitstream/handle/1721.1/155928/3643562.3672612.pdf?sequence=1


def plotAccuracy(priceList ):
    means = []
    uppers, lowers = [],[]
    upper2sigs, lower2sigs = [],[]
    timestamps = [x for x in range(0,maxMarketLength,10)]
    #timestamps are 600s apart
    for time in timestamps:
        prices = [price[time] for price in priceList if time<len(price)]
        means.append(np.mean(prices))

        lower,upper = np.quantile(prices,[0.16,0.84])-means[-1]
        lower2sig,upper2sig = np.quantile(prices,[0.022,0.978])-means[-1]
        uppers.append(upper)
        lowers.append(lower)
        upper2sigs.append(upper2sig)
        lower2sigs.append(lower2sig)

    errors = np.abs(np.vstack((np.array(lowers),np.array(uppers))))

    error2sig = np.abs(np.vstack((np.array(lower2sigs),np.array(upper2sigs))))
    times = [x*1/6 for x in timestamps]
    #plt.scatter(times,means)
    #plt.errorbar(times,means,yerr=errors,color='cyan')

    plt.plot(times,means,color='green',label=' Mean')
    plt.fill_between(times,means-errors[0],means+errors[1],color='green',alpha=0.3,label="$\pm1\sigma$")
    plt.fill_between(times,means-error2sig[0],means+error2sig[1],color='green',alpha=0.3,label=r"$\pm2\sigma$")
    leg = plt.legend()
    leg.legend_handles[1].set_alpha(0.6)
    plt.ylim(0,1)
    plt.xlabel("Time before market resolution (hours)")
    plt.ylabel("Price")
    plt.savefig("plots/accuracy.png")
    #plt.show()
    plt.clf()

def plotAccuracyRelative(priceList ):
    means = []
    uppers, lowers = [],[]
    upper2sigs, lower2sigs = [],[]
    resolution=100
    timestamps = [x/resolution for x in range(0,resolution)]
    print(timestamps)
    
    for time in timestamps:
        prices = [price[min(round(time*len(price)),len(price)-1)] for price in priceList]
        means.append(np.mean(prices))

        lower,upper = np.quantile(prices,[0.16,0.84])-means[-1]
        lower2sig,upper2sig = np.quantile(prices,[0.022,0.978])-means[-1]
        uppers.append(upper)
        lowers.append(lower)
        upper2sigs.append(upper2sig)
        lower2sigs.append(lower2sig)

    errors = np.abs(np.vstack((np.array(lowers),np.array(uppers))))

    error2sig = np.abs(np.vstack((np.array(lower2sigs),np.array(upper2sigs))))

    plt.plot(timestamps,means,color='green',label=' Mean')
    plt.fill_between(timestamps,means-errors[0],means+errors[1],color='green',alpha=0.3,label="$\pm1\sigma$")
    plt.fill_between(timestamps,means-error2sig[0],means+error2sig[1],color='green',alpha=0.3,label=r"$\pm2\sigma$")
    leg = plt.legend()
    leg.legend_handles[1].set_alpha(0.6)
    plt.ylim(0,1)
    plt.xlabel("Time before market resolves (%)")
    plt.ylabel("Price")
    plt.savefig("plots/accuracy_relative.png")
    #plt.show()
    plt.clf()

plotAccuracy(priceList)
plotAccuracyRelative(priceList)

#does volume have an impact? brier score (MSE) vs volume
#maybe weight above plots by volume?

#volume follows log-normal
plt.hist(np.log(volumeList),bins=100,histtype='step')
plt.xlabel("log(volume)")
plt.savefig("plots/volume.png")
#plt.show()
plt.clf()

brierPerMarket = []
BCE = [] 
for price in priceList:
    brierPerMarket.append(float(((1-np.array(price))**2).sum()/len(price)))
    BCE.append( -np.log(price).mean())

plt.hist(brierPerMarket,bins=100,histtype='step')
plt.savefig("plots/brier.png")
plt.clf()


plt.scatter(np.log(volumeList),BCE)
plt.yscale("log")
#plt.scatter(volumeList,brierPerMarket)
plt.show()
#need to see quantiles of this to assess effect
