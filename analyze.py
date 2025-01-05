import json
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
import scipy


def flatten(listOfLists,maxCutoff=1):
    return [e for l in listOfLists for e in l[:int(maxCutoff*len(l))]]

def getCalibration(priceList,weighted=False):
    if not weighted:
        h,bins=np.histogram(flatten(priceList),range=(0,1),bins=51)
    else:
        weights = [[1/len(x)]*len(x) for x in priceList]
        h,bins=np.histogram(flatten(priceList),range=(0,1),bins=51,weights=flatten(weights))
    midpoints = 0.5*(bins[:-1]+bins[1:])
    ratios = h/(h+h[::-1])
    return ratios,midpoints

def computeWeightedAccuracy(priceList,maxCutoff=1.0):
    #weight markets by inverse length
    weights = [[1/len(x)]*len(x) for x in priceList]
    h,bins=np.histogram(flatten(priceList,maxCutoff=maxCutoff),range=(0,1),bins=51,weights=flatten(weights,maxCutoff=maxCutoff))
    midpoints = 0.5*(bins[:-1]+bins[1:])
    ratios = h/(h+h[::-1])
    return ratios,midpoints


def bootstrapCalibration(priceList, draws, weighted=False,maxCutoff=1.0):
    N = len(priceList)
    if not weighted: 
        ratios, midpoints = getCalibration(priceList)
    else:
        ratios, midpoints = getCalibration(priceList,weighted=True)
    bootstraps = np.zeros((draws,len(midpoints)))

    block = 100
    for i in range(0,draws,block):
        print(i)
        #choose indices of markets
        sampleIdxs = np.random.randint(0,N,size=(min(block,draws-i),N))
        
        for j,sampleIdx in enumerate(sampleIdxs):
            sample = [priceList[idx] for idx in sampleIdx]
            if not weighted:
                ratios_draw,_ = getCalibration(sample)
            else:
                ratios_draw,_ = getCalibration(sample,weighted=True)
            bootstraps[i+j] = ratios_draw
    
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

def plotPricesAbsoluteTime(priceList):
    means = []
    uppers, lowers = [],[]
    upper2sigs, lower2sigs = [],[]
    maxMarketLength = max([len(market) for market in priceList])
    timestamps = [x for x in range(0,maxMarketLength,10)]
    #timestamps are 10m (600s) apart
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
    times = [x*1/6 for x in timestamps] #in minutes

    plt.plot(times,means,color='green',label=' Mean')
    plt.fill_between(times,means-errors[0],means+errors[1],color='green',alpha=0.3,label="$\pm1\sigma$")
    plt.fill_between(times,means-error2sig[0],means+error2sig[1],color='green',alpha=0.3,label=r"$\pm2\sigma$")
    leg = plt.legend()
    leg.legend_handles[1].set_alpha(0.6)
    plt.ylim(0,1)
    plt.xlabel("Time before market resolution (hours)")
    plt.ylabel("Price")
    plt.savefig("plots/pricesAbsolute.png")
    plt.clf()

def plotPricesRelativeTime(priceList,resolution=100 ):
    means = []
    uppers, lowers = [],[]
    upper2sigs, lower2sigs = [],[]
    timestamps = [x/resolution for x in range(0,resolution)]
    
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
    plt.clf()

def flattenCompliment(listOfLists):
    return [1-e for l in listOfLists for e in l]

def getBrierDecomposition(priceList):
    #brier score is invariant to binary labeling.
    #to do decomposition, randomly pick labeling
    
    random.shuffle(priceList)
    P1 = priceList[0:len(priceList)//2]
    P2 = priceList[len(priceList)//2:-1]
    
    #take P1 as is, resolved "true"
    #take 1-P2, resolved "false"

    #weight markets by inverse length
    w1, w2 = [[1/len(x)]*len(x) for x in P1],[[1/len(x)]*len(x) for x in P2]
    h1,bins=np.histogram(flatten(P1),range=(0,1),bins=51,weights=flatten(w1))
    
    h2,bins=np.histogram(flattenCompliment(P2),range=(0,1),bins=51,weights=flatten(w2))
    midpoints = 0.5*(bins[:-1]+bins[1:])
    P_Y_cond_p = h1/(h1+h2)
    P_Y = sum(h1)/sum(h1+h2)

    z1=(P_Y_cond_p-midpoints)**2*h1
    z2=(h2/(h1+h2)-midpoints[::-1])**2*h2[::-1]

    CAL = sum(z1+z2 )/sum(h1+h2)

    SHP = sum(  ( (P_Y_cond_p-sum(h1)/sum(h1+h2))**2*h1 + (h2/(h1+h2)-sum(h2)/sum(h1+h2))**2*h2[::-1] )/sum(h1+h2) )
    RNG = P_Y*(1-P_Y)
    return CAL,SHP,RNG

class PredictionMetrics:
    def __init__(self, fname):
        with open(fname,"r") as f:
            self.data = json.load(f)
            counter=0
            for i,market in enumerate(self.data):
                if "prices" not in market.keys():
                    market["prices"] = []
                    market["result"] = "-1"

                elif len(market["prices"])>0:
                    counter+=1
            print("Found {} markets with price data in '{}'".format(counter, fname))
            self.priceList = self.cleanPriceLists()
    
    def cleanPriceLists(self):
        priceList = []
        for market in self.data:
            if len(market["prices"])>0 and "volumeNum" in market and market["volumeNum"]>0:
                listOfPrices = [p["p"] for p in market["prices"]]
                priceList.append(listOfPrices[::-1])
        return priceList
    
    def findWorstSuprises(self,lowestPrice=0.03,medianCutoff=0.1):
        for market in self.data:
            if len(market["prices"])>0 and "volumeNum" in market and market["volumeNum"]>0:
                listOfPrices = [p["p"] for p in market["prices"]]
                volume = float(market["volume"])
                duration = len(listOfPrices)*1/6
                median = sorted(listOfPrices)[len(listOfPrices)//2]
                if min(listOfPrices)<lowestPrice or median<medianCutoff :
                    print(" ".join([market["outcomes"],market["result"],str(market["prices"][-1]['p']),market["outcomePrices"],market["slug"]]))
                    times = [datetime.datetime.utcfromtimestamp(t['t']) for t in market['prices']]

                    plt.plot(times,listOfPrices)
                    plt.xlabel("Time")
                    plt.ylabel("Price")
                    plt.gcf().autofmt_xdate()
                    plt.title(market['slug']+"\n min prob: {:.3f}     median prob: {:.3f}".format(min(listOfPrices),median))
                    plt.savefig("plots/folly_{}.png".format(market['slug']))
                    plt.clf()

    def getBrierScoreMetrics(self):
        brierScore = getBrierScore(self.priceList)
        brierScoreProb = (1-brierScore**0.5)
        brierScoreWeighted = getBrierScore(self.priceList,True)
        brierScoreProbWeighted = (1-brierScoreWeighted**0.5)
        print("Brier score (unweighted): {:.4f}\t Prob: {:.4f}".format(brierScore,brierScoreProb))
        print("Brier score (weighted): {:.4f}\t Prob: {:.4f}".format(brierScoreWeighted,brierScoreProbWeighted))

        brierPerMarket = []
        for price in self.priceList:
            brierPerMarket.append(float(((1-np.array(price))**2).sum()/len(price)))
        plt.hist(brierPerMarket,bins=100,histtype='step')
        plt.xlabel("Market Brier Score (weighted)")
        plt.savefig("plots/brier.png")
        plt.clf()

    def getCalibrationMetrics(self,trials=500):
        ratios,midpoints = getCalibration(self.priceList) 
        print("bootstrapping calibration with {} draws".format(trials))
        errors = bootstrapCalibration(self.priceList,trials)

        plt.plot(midpoints,ratios)
        plt.fill_between(midpoints,ratios-errors[0],ratios+errors[1],color='blue',alpha=0.3,label="$\pm1\sigma$")

        plt.plot([0,1],[0,1],linestyle='dashed',color='green')
        plt.title("Polymarket Calibration")
        plt.xlabel("Market Price")
        plt.ylabel("Outcome Frequency")
        plt.savefig("plots/calibration_unweighted.png")
        plt.clf()


        # calibration weighted by inverse market duration
        ratios,midpoints = getCalibration(self.priceList,weighted=True)
        errors = bootstrapCalibration(self.priceList,trials,weighted=True)

        plt.plot(midpoints,ratios,label="Observed")
        plt.fill_between(midpoints,ratios-errors[0],ratios+errors[1],color='blue',alpha=0.3,label="$\pm1\sigma$ CI")

        plt.plot([0,1],[0,1],linestyle='dashed',color='green',label="Ideal")
        plt.legend(loc="upper left")
        plt.title("Polymarket Calibration")
        plt.xlabel("Market Price")
        plt.ylabel("Outcome Frequency")
        plt.savefig("plots/calibration_weighted.png")
        #plt.show()
        plt.clf()

    def plotPricesOverTime(self):
        #prices over time
        plotPricesAbsoluteTime(self.priceList)
        plotPricesRelativeTime(self.priceList)

    def getBrierDecompositionMetrics(self,trials=100):
        #due to randomness in labeling, average results over trials
        cals=[]
        shps=[]
        rngs=[]
        for i in range(trials):
            cal,shp,rng = getBrierDecomposition(self.priceList)
            cals.append(cal) 
            shps.append(shp)
            rngs.append(rng)
        CAL=sum(cals)/trials
        SHP=sum(shps)/trials
        RNG=sum(rngs)/trials
        print("Computing Brier score decomposition over {} trials".format(trials))
        print("CAL: {:.3f}\tSHP: {:.3f},\tRNG: {:.3f}\t Brier: {:.3f}".format(CAL,SHP,RNG,CAL-SHP+RNG))

if __name__=="__main__":
    predictions = PredictionMetrics("marketsWithPrices.json")
    predictions.findWorstSuprises(lowestPrice=0.02,medianCutoff=0.05)
    predictions.getBrierScoreMetrics()
    predictions.getCalibrationMetrics(trials=500)
    predictions.getBrierDecompositionMetrics(trials=100)
    predictions.plotPricesOverTime()


