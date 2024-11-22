import json
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
import matplotlib.dates as mdates
import pdb
import scipy

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
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.gcf().autofmt_xdate()
            plt.title(market['slug']+"\n min prob: {:.3f}     median prob: {:.3f}".format(min(listOfPrices),median))
            plt.savefig("plots/folly_{}.png".format(market['slug']))
            #plt.show()
            #pdb.set_trace()
            plt.clf()

        #Too many! 
        #if  median>0.95:
        #    print(" ".join([market["outcomes"],market["result"],str(market["prices"][-1]['p']),market["outcomePrices"],market["slug"]]))
        #    times = [datetime.datetime.utcfromtimestamp(t['t']) for t in market['prices']]

        #    plt.plot(times,listOfPrices)
        #    plt.gcf().autofmt_xdate()
        #    plt.title(market['slug']+"\n min prob: {:.3f}     median prob: {:.3f}".format(min(listOfPrices),median))
        #    plt.savefig("plots/glory_{}.png".format(market['slug']))
        #    #plt.show()
        #    #pdb.set_trace()
        #    plt.clf()

def flatten(listOfLists):
    return [e for l in listOfLists for e in l]

def computeAccuracy(prices):
    h,bins=np.histogram(prices,range=(0,1),bins=51)
    midpoints = 0.5*(bins[:-1]+bins[1:])
    ratios = h/(h+h[::-1])
    return ratios,midpoints

def computeWeightedAccuracy(priceList):
    #weight markets by inverse length
    weights = [[1/len(x)]*len(x) for x in priceList]
    h,bins=np.histogram(flatten(priceList),range=(0,1),bins=51,weights=flatten(weights))
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
    return np.abs(np.vstack((lower,upper)))


def quantileBootstrapMarkets(priceList, draws, weightByLength=False):
    N = len(priceList)
    if not weightByLength: 
        ratios, midpoints = computeAccuracy(flatten(priceList))
    else:
        ratios, midpoints = computeWeightedAccuracy(priceList)
    bootstraps = np.zeros((draws,len(midpoints)))

    block = 100
    for i in range(0,draws,block):
        print(i)
        #samples = np.random.choice(data,size=(min(block,draws-i),N))

        #choose indices of markets
        sampleIdxs = np.random.randint(0,N,size=(min(block,draws-i),N))
        
        for j,sampleIdx in enumerate(sampleIdxs):
            sample = [priceList[idx] for idx in sampleIdx]
            if not weightByLength:
                ratios_draw,_ = computeAccuracy(flatten(sample))
            else:
                ratios_draw,_ = computeWeightedAccuracy(sample)

            bootstraps[i+j] = ratios_draw
            #sum_squared_variation += (ratios_draw-ratios)**2
    
    lower,upper = np.quantile(bootstraps,[0.16,0.84],axis=0)-ratios
    return np.abs(np.vstack((lower,upper)))


def getBrierScore(priceList,weighted=False):
    if not weighted:
        prices = np.array(flatten(priceList))
        return ((prices-1)**2).mean()
    else:
        #weight each market by inverse of length
        MSE=0
        for prices in priceList:
            prices = np.array(prices)
            MSE+= ((prices-1)**2).mean()
        return MSE/len(priceList)

print("longest market: {}".format(maxMarketLength))

brierScore = getBrierScore(priceList)
brierScoreProb = (1-brierScore**0.5)
print("Brier score: {:.4f}\t Prob: {:.4f}".format(brierScore,brierScoreProb))
brierScoreWeighted = getBrierScore(priceList,True)
brierScoreProbWeighted = (1-brierScoreWeighted**0.5)
print("Brier score (weighted): {:.4f}\t Prob: {:.4f}".format(brierScoreWeighted,brierScoreProbWeighted))

#plt.hist(flatten(priceList),range=(0,1),bins=100,histtype='step')
#plt.xlabel("Price")
#plt.savefig("plots/prices.png")
#plt.clf()
#
#plt.hist(flatten(priceList),weights=flatten([[1/len(x)]*len(x) for x in priceList]) ,range=(0,1),bins=100,histtype='step')
#plt.xlabel("Price")
#plt.savefig("plots/prices_weighted.png")
#plt.clf()

#TODO?: percentile bootstrap to BCa file:///home/nick/Downloads/casi_corrected_03052021.pdf#page=212
#t_dot is average of statistic over jackknife samples
#a=1/6*(sum( (t_i - t_dot )^3) )/(sum((t_i-t_dot)^2)^1.5
#G(alpha) is proportion of bootstrap samples below alpha
#z_0 = Phi^-1( p0 ) i.e. p0 is proportion of bootstrapped samples below original estimate t
#t_bca = G^-1 ( Phi( z_0 + (z_0 + z_alpha)/(1-a(z_0+z_alpha))))

N_bootstraps=1
ratios,midpoints = computeAccuracy(prices)
errors = quantileBootstrapMarkets(priceList,N_bootstraps)
#plt.errorbar(midpoints,ratios,yerr=errors)

plt.plot(midpoints,ratios)
plt.fill_between(midpoints,ratios-errors[0],ratios+errors[1],color='blue',alpha=0.3,label="$\pm1\sigma$")

plt.plot([0,1],[0,1],linestyle='dashed',color='green')
plt.title("Polymarket Calibration")
plt.xlabel("Market Price")
plt.ylabel("Outcome Frequency")
plt.savefig("plots/calibration_unweighted.png")
plt.clf()

plt.hist(durations,histtype='step',bins=100)
plt.xlabel("Market duration (h)")
plt.savefig("plots/duration.png")
##plt.show()
plt.clf()


# calibration weighted by inverse market duration
ratios,midpoints = computeWeightedAccuracy(priceList)
errors = quantileBootstrapMarkets(priceList,N_bootstraps,weightByLength=True)

plt.plot(midpoints,ratios)
plt.fill_between(midpoints,ratios-errors[0],ratios+errors[1],color='blue',alpha=0.3,label="$\pm1\sigma$")

plt.plot([0,1],[0,1],linestyle='dashed',color='green')
plt.title("Polymarket Calibration")
plt.xlabel("Market Price")
plt.ylabel("Outcome Frequency")
plt.savefig("plots/calibration_weightByLength.png")
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

    plt.plot(timestamps[::-1],means,color='green',label=' Mean')
    plt.fill_between(timestamps[::-1],means-errors[0],means+errors[1],color='green',alpha=0.3,label="$\pm1\sigma$")
    plt.fill_between(timestamps[::-1],means-error2sig[0],means+error2sig[1],color='green',alpha=0.3,label=r"$\pm2\sigma$")
    leg = plt.legend()
    leg.legend_handles[1].set_alpha(0.6)
    plt.margins(x=0)
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

plt.hist(BCE,bins=100,histtype='step')
plt.savefig("plots/BCE.png")
plt.clf()



class Percentile:
    def __init__(self,percentile):
        self.percentile = percentile
    def __call__(self,values):
        return np.quantile(values,self.percentile) 

def upperQuantile(values):
    return np.quantile(values,0.84)

#bins=30
#bins=[-1,0,1,2]+list(np.linspace(3,15,13))+[16,20,22]
bins = list(range(-1,23))
#metric = BCE
metric = brierPerMarket
means,_,_ = scipy.stats.binned_statistic(np.log(volumeList),metric,statistic='mean',bins=bins)
upper,edges,_ = scipy.stats.binned_statistic(np.log(volumeList),metric,statistic=Percentile(0.84),bins=bins)
lower,_,_ = scipy.stats.binned_statistic(np.log(volumeList),metric,statistic=Percentile(0.16),bins=bins)


midpoints =  0.5*(edges[:-1]+edges[1:])
plt.scatter(np.log(volumeList),metric,color="salmon")
plt.xlabel("log(Market Volume)")
plt.ylabel("Market Brier Score")
plt.plot(midpoints,means,color='maroon',label="Mean")

#plt.fill_between(midpoints,lower,upper,color='violet',alpha=0.3,label=r"$\pm1\sigma$")

#m,b = np.polyfit(np.log(volumeList),BCE,1)
m,b = np.polyfit(np.log(volumeList),metric,1)
plt.plot(midpoints,midpoints*m+b,color="dodgerblue",label="{:.3f}x+{:.3f}".format(m,b))
plt.legend()
#plt.yscale("log")
#plt.scatter(volumeList,brierPerMarket)
#print("Doubling volume corresponds to {:.3f} more bits".format(m))
#plt.savefig("plots/WBSvslogV.png")
plt.savefig("plots/brierPerMarketvslogV.png")
plt.show()

#need to see quantiles of this to assess effect
