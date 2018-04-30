import RMS_Cats as cats
import pandas as pd
import numpy as np
import logging
import logging.config
import json
import os

def setup_logging(default_path = 'log_config.json',default_level = logging.INFO,env_key='LOG_CFG'):
    """Setup logging configureation from JSON file
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)


def main():
    """main entry point to run the cat models
    """

    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Program Started")
    

    catfiles = {'HR':{'file':r'Y:\ECM\2018\Data\RMSv17_ELTs\Prosp_HUNT_byStateLOB_NetPreCat_ELT.csv','run':True},
    'EQ':{'file':r'Y:\ECM\2018\Data\RMSv17_ELTs\Prosp_EQFFSL_byStateLOB_NetPreCat_ELT.csv','run':True},
    'OW':{'file':r'Y:\ECM\2018\Data\RMSv17_ELTs\Prosp_OWLow_byStateLOB_Gross_ELT.csv','run':False},
    'WN':{'file':r'Y:\ECM\2018\Data\RMSv17_ELTs\Prosp_WN_byStateLOB_NetPreCat_ELT.csv', 'run':False}
    }
    sims = 500

    writer = pd.ExcelWriter('cat_rms_results.xlsx', engine='openpyxl')
    logger.info(f'number of sims {sims}')
    
    for peril in catfiles:
        
        if catfiles[peril]['run']:
            logger.info(f'{peril} running')
            pathfile = catfiles[peril]['file']
            rmsdata = cats.load_rms_file(pathfile)
            output = cats.sim_rms_results(rmsdata, sims)
            #output.to_csv(f"{peril}_output.csv")
            oepdf = cats.calc_total_oep_curve(output, sims)
            oepdf.to_excel(excel_writer = writer, sheet_name=peril, startcol=1)
            aepdf = cats.calc_total_aep_curve(output, sims)
            aepdf.to_excel(excel_writer = writer, sheet_name=peril, startcol=5)
            logger.info(f'{peril} Complete')
        else:
            logger.info(f'{peril} skipped')
    
    
    
    logger.info("Cat run complete")
    writer.save()

if __name__=="__main__":
    main()