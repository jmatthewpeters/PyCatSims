import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import scipy.stats as stats
import logging
import sqlite3
import dask.dataframe as dd


#from numba import jit


def aggregate_rms_data(rmsdata):
    logger = logging.getLogger(__name__)
    logger.info("start aggregate rms data")
    rmsdata['stddevi'] = rmsdata['stddevi']**2
    rmsdata['state'] = 'total'
    rmsdata['line'] = 'totalline'
    aggdata = rmsdata.groupby(['eventid', 'rate', 'perspcode', 'state', 'line']).sum()
    aggdata['stddevi'] = np.sqrt(aggdata['stddevi'])
    logger.info("finished aggregating rmsdata")
    return aggdata

def output_igloodata(rmsdata):
    logger = logging.getLogger(__name__)
    logger.info("outputing igloo data")
    lines = pd.DataFrame(rmsdata["line"].unique(), columns=["line"])
    #states = pd.DataFrame(rmsdata["state"].unique(), columns=["state"])

    outfile = path.normpath(r'Y:\ECM\2018\Data\aggregated_HU.csv')
    igloodata = rmsdata.pivot_table(index = ['eventid', 'rate', 'perspcode'], columns=['state', 'line'], aggfunc=np.sum)
    igloodata = igloodata.fillna(0)
    igloodata = igloodata.reset_index()

    tmp = pd.DataFrame()
    tmp['eventid'] = igloodata['eventid']
    tmp['rate'] = igloodata['rate']

    for line in lines['line']:
        tmp[f'mean-{line}'] = igloodata[('meanvalue', 'total', line)]
        tmp[f'stddevc-{line}'] = igloodata[('stddevc', 'total', line)]
        tmp[f'stddevi-{line}'] = igloodata[('stddevi', 'total', line)]
        tmp[f'exposure-{line}'] = igloodata[('exposure', 'total', line)]

    tmp.to_csv(outfile)



def simulate_events(numberofevents, eventlookup, sims):
    '''final output {sim, event, eventid}'''
    logger = logging.getLogger(__name__)
    logger.info('start simulation of events')
    totalevents = numberofevents['events'].sum()
    catevents = np.empty((max(totalevents, sims), 3))
    rollingevent = 1
    
    eventids = eventlookup.index
    normalizedprob = eventlookup['rate']/eventlookup['rate'].sum()
    
    for sim in range(1, sims+1):
        events = numberofevents['events'][sim]

        lowerindex = rollingevent - 1 
        upperindex = rollingevent + events - 1
        #data = np.empty((events, 3))
        #simindex = pd.Multiindex([np.full(events, sim),np.arange(1, events+1,1)], names=['sim', 'event'] )
        simindex = np.full(events, int(sim))
        simindex.astype(int)
        #data[0:events, 0] = simindex
        catevents[lowerindex:upperindex, 0] = simindex
        eventindex = np.arange(1, events+1,1)
        eventindex.astype(int)
        catevents[lowerindex:upperindex, 1] = eventindex
        #data[0:events, 1] = eventindex
        eventlookups = np.random.choice(eventids, p=normalizedprob, size=events)
        eventlookups.astype(int)
        catevents[lowerindex:upperindex, 2] = eventlookups
        #data[0:events, 2] = eventids
        #catevents = np.append(catevents,  data, axis=0)
                
        rollingevent = rollingevent + events
                
        
    catevents = catevents.astype(int)
    catdatapd = dd(catevents, columns=['simulation', 'eventseq', 'eventid'])
    catdatapd['rand'] = np.random.uniform(0,1, size=max(totalevents, sims))
    #catdatapd = catdatapd.join(eventlookup, on='eventlookup')
    logger.info('finished generating simulated events')
    return catdatapd


def simulate_state_lines_losses(eventlookup, freq_mean, states, lines, sims):
    '''assembles state line level events based on the year event
    '''
    logger = logging.getLogger(__name__)
    logger.info('start state lines losses')
    numberofevents = dd(np.random.poisson(freq_mean, sims), index=np.arange(1, sims+1), columns=['events'])
    catevents = simulate_events(numberofevents, eventlookup, sims)

    simsevents = list(range(len(catevents)))
    #combinedResults = xr.DataArray(np.empty((len(states), len(lines), len(catevents), 4)),name="catevents", coords=[states['state'], lines['line'], simsevents, ["sim", "eventseq", "eventid", "rand"]], dims=['state', 'line', 'eventsim', 'data'] )

    logger.info('start to build full array of losses, combining state lines with events')
    sim_events = dd()
    firstloop = True
    for state in states['state']:
        print(f'start {state}')
        for line in lines['line']:
            #combinedResults.loc[state, line] = catevents.copy()
            print(f'start {line}')
            b = catevents.copy()
            b['state'] = state
            b['line'] = line
            if firstloop: 
                sim_events = b
                firstloop = False
            else: 
                sim_events = dd.concat([sim_events, b])


    #sim_events = pd.concat(a, ignore_index=True, axis=0, )
    logger.info('Completed combined state lines with events')
    return combinedResults


def calculate_rms_loss(df, dist):
    ''' calculcates the rms loss based on the dataframe of cat data
    '''
    logger = logging.getLogger(__name__)
    logger.info("Calculating losses")
    tolerance = 1e-6
    claimSD = df['stddevc'] + df['stddevi']
    exposure = df['exposure']
    meanvalue = df['meanvalue']
    rand = df['rand']
    
    if dist == "Beta":
        damage_ratio_mean = np.where(exposure < tolerance, 0, meanvalue / exposure)
        damage_ratio_cv = np.where(meanvalue < tolerance, 0, claimSD / meanvalue )
        SU_Alpha = np.where(damage_ratio_cv <= 0, 0,  ((1 - damage_ratio_mean)/ damage_ratio_cv ** 2 - damage_ratio_mean))
        SU_Beta =  np.where(damage_ratio_mean < tolerance, 0, SU_Alpha * (1 - damage_ratio_mean)/ damage_ratio_mean)
        
        return np.where(np.minimum(SU_Alpha, SU_Beta)>0, exposure * stats.beta(SU_Alpha, SU_Beta).ppf(rand),  meanvalue)
        
    elif dist == "Lognormal":
        lmean = np.where(meanvalue == 0,0, np.log(meanvalue) - 0.5 * np.log(1 + (claimSD / meanvalue) ** 2))
        lstdev = np.where(meanvalue == 0, 0, np.sqrt(np.log(1 + (claimSD / meanvalue) ** 2)))
        
        return np.where(np.minimum(meanvalue, claimSD)>0, stats.lognorm(lmean, lstdev).ppf(rand), meanvalue)
    
    else:
        #normal
        return np.maximum(0, stats.norm(meanvalue, claimSD).ppf(rand))
    
def create_db():
    conn = sqlite3.connect('cats.db')



def sim_rms_results(rmsdata, sims = 100):
    '''simulates the RMS results based on the 
    '''
    logger = logging.getLogger(__name__)
    logger.info('set lines')
    lines = dd.from_array(rmsdata["line"].unique(), columns=["line"])
    logger.info('set states')
    states = dd.from_array(rmsdata["state"].unique(), columns=["state"])
    #rmsdf = pd.pivot_table(rmsdata, values=['PERSPVALUE', 'STDDEVI', 'STDDEVC', 'EXPVALUE'], index=['LOBNAME', 'STATE', 'EVENTID'])
    logger.info('pivot rms data for events')
    rmsevents = pd.pivot_table(rmsdata, values='rate', index='eventid')
    freq_mean = rmsevents.sum()
    #EVENTID 	RATE 	PERSPCODE 	STATE 	LOBNAME 	PERSPVALUE 	STDDEVI 	STDDEVC 	EXPVALUE
    #catindex = [rmsdata['LOBNAME'].values, rmsdata['STATE'].values, rmsdata['EVENTID'].values]
    logger.info('create rmslookup table')
    rmslookup = dd.from_array({'line':rmsdata['line'].values, 'state':rmsdata['state'].values, 'eventid':rmsdata['eventid'].values, 
                              'meanvalue':rmsdata['meanvalue'].values,'stddevi':rmsdata['stddevi'].values, 
                              'stddevc':rmsdata['stddevc'].values, 'exposure':rmsdata['exposure'].values })
    #rmslookup.index.names = ['lob', 'state', 'eventid']
    #eventlookup = pd.DataFrame(rmsevents.index)

    simulated_events = simulate_state_lines_losses(rmsevents, freq_mean, states, lines, sims) 
    simulated_events = simulated_events.to_dataframe()
    simulated_cats = dd.merge(simulated_events, rmslookup, how='inner', left_on=['line', 'state', 'eventid'], right_on=[ 'line', 'state', 'eventid'])
    #simulated_cats['rand']= np.random.uniform(0,1, len(simulated_cats))
    simulated_cats['loss'] = calculate_rms_loss(simulated_cats, "Beta")
    logger.info("set index to simulated cats")
    simulated_cats = simulated_cats.set_index(['line', 'state','simulation', 'eventid', 'eventseq' ])
    
    #results = results.reset_index()
    logger.info("Completed simulating cats")
    return simulated_cats

def load_rms_file(catfilepath):
    logger = logging.getLogger(__name__)
    logger.info("start file load process")
    rmsfile = path.normpath(catfilepath)

    rmsdatatypes = {"EVENTID":np.int64,"RATE":np.float64,"PERSPCODE":str,"STATE":str,"LOBNAME":str,"PERSPVALUE":np.float64,
                    "STDDEVI":np.float64,"STDDEVC":np.float64, "EXPVALUE":np.float64}
    rmsdata = dd.read_csv(rmsfile, dtype=rmsdatatypes)
    rmsdata = rmsdata.rename(columns={"EVENTID":'eventid',"RATE":'rate',"PERSPCODE":'perspcode',"STATE":'state',"LOBNAME":'line',
                "PERSPVALUE":'meanvalue',"STDDEVI":'stddevi',"STDDEVC":'stddevc', "EXPVALUE":'exposure'} )
    rmsdata['line'] = rmsdata['line'].str.title()
    logger.info("completed file load process")
    return rmsdata


def calc_total_oep_curve(catlosses, simnumber):
    '''takes the simulatied losses and calcs the OEP curve for the events
    '''
    logger = logging.getLogger(__name__)
    logger.info("Calulating OEP results")
    totallossbyevent = pd.DataFrame(catlosses['loss'].groupby(level = ['simulation', 'eventseq', 'eventid']).sum())
    lossmax = totallossbyevent.groupby(level = 'simulation').max()
    simsarray = pd.DataFrame(np.arange(1,simnumber+1), columns=['simulation'])
    catlosses = simsarray.merge(lossmax, how='left', left_on='simulation', right_index=True)
    catlosses = catlosses.fillna(0)
    #catloss = catloss.set_index("simulation")

    quantiles = {'99.9th':{'ReturnPeriod':1000, 'quantile':.999},
                '99.8th':{'ReturnPeriod':500, 'quantile':.998},
                '99.6th':{'ReturnPeriod':250, 'quantile':.996},
                '99.5th':{'ReturnPeriod':200, 'quantile':.995},
                '99.0th':{'ReturnPeriod':100, 'quantile':.99},
                '98.0th':{'ReturnPeriod':50, 'quantile':.98},
                '96.0th':{'ReturnPeriod':25, 'quantile':.96},
                '95.0th':{'ReturnPeriod':20, 'quantile':.95},
                '90.0th':{'ReturnPeriod':20, 'quantile':.90},
                }
    output = {}
    for quant in quantiles:
        output[quant] = catlosses['loss'].quantile(quantiles[quant]['quantile'])
    output['mean'] = catlosses['loss'].mean()
    s = pd.Series(output, name='OEP')
    s.index.name = 'Quantile'
    output = pd.DataFrame(s)
    return output



def calc_total_aep_curve(catlosses, simnumber):
    '''takes the simulatied losses and calcs the aep curve
    '''
    logger = logging.getLogger(__name__)
    logger.info("Calulating AEP results")
    catlosses = pd.DataFrame(catlosses['loss'].groupby(level = ['simulation']).sum())
    simsarray = pd.DataFrame(np.arange(1, simnumber+1), columns=['simulation'])
    catlosses = simsarray.merge(catlosses, how='left', left_on='simulation', right_index=True)
    catlosses = catlosses.fillna(0)
    #catloss = catloss.set_index("simulation")

    quantiles = {'99.9th':{'ReturnPeriod':1000, 'quantile':.999},
                '99.8th':{'ReturnPeriod':500, 'quantile':.998},
                '99.6th':{'ReturnPeriod':250, 'quantile':.996},
                '99.5th':{'ReturnPeriod':200, 'quantile':.995},
                '99.0th':{'ReturnPeriod':100, 'quantile':.99},
                '98.0th':{'ReturnPeriod':50, 'quantile':.98},
                '96.0th':{'ReturnPeriod':25, 'quantile':.96},
                '95.0th':{'ReturnPeriod':20, 'quantile':.95},
                '90.0th':{'ReturnPeriod':20, 'quantile':.90},
                }
    output = {}
    for quant in quantiles:
        quantile = catlosses['loss'].quantile(quantiles[quant]['quantile'])
        output[quant] = quantile

    output['mean'] = catlosses['loss'].mean()

    s = pd.Series(output, name='AEP')
    s.index.name = 'Quantile'
    output = pd.DataFrame(s)
    return output


if __name__ == "__main__":
    rmsfile = path.normpath(r'Y:\ECM\2018\Data\RMSv17_ELTs\__GuideOne_Core_Promont_Catalytic_Prosp_HUNT_byStateLOB_NetPreCat_ELT.csv')
    #rmsfile = path.normpath(r'Y:\ECM\2018\Data\TestRMS.csv')
    rmsdatatypes = {"EVENTID":np.int64,"RATE":np.float64,"PERSPCODE":str,"STATE":str,"LOBNAME":str,"PERSPVALUE":np.float64,
                    "STDDEVI":np.float64,"STDDEVC":np.float64, "EXPVALUE":np.float64}
    rmsdata = pd.read_csv(rmsfile, dtype=rmsdatatypes)
    rmsdata = rmsdata.rename(index=str,columns={"EVENTID":'eventid',"RATE":'rate',"PERSPCODE":'perspcode',"STATE":'state',"LOBNAME":'line',
                "PERSPVALUE":'meanvalue',"STDDEVI":'stddevi',"STDDEVC":'stddevc', "EXPVALUE":'exposure'} )
    rmsdata['line'] = rmsdata['line'].str.title()
    sims = 10000
    output = sim_rms_results(rmsdata, sims=sims)