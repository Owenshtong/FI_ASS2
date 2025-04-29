import numpy as np
import pandas as pd
import wrds
from datetime import datetime
from dateutil.relativedelta import relativedelta


conn = wrds.Connection(wrds_username='owen_tong')


# Define SQL query to retrieve daily price and shares outstanding
def quote_equity_CRSP(s0, sT, permno):
    query = "SELECT date, permno, prc, shrout " +\
            "FROM crsp.dsf " +\
            "WHERE permno = " + permno +\
            " AND date BETWEEN " + str("'" + s0 +"'" + " AND " + "'" + sT +"'") +\
            " ORDER BY date"
    S = conn.raw_sql(query)
    S["shrout"] = S["shrout"] * 1000
    S["Mrk_cap"] = S["prc"] * S["shrout"]

    S.index = pd.to_datetime(S.date)
    S = S.drop(["date", "permno"], axis = 1)
    return S


def quote_debt_compstat(s0, sT, gvkey):
    # Shift the starting time for 4 months
    s0_quater = datetime.strptime(s0, "%Y-%m-%d")
    s0_quater = s0_quater - relativedelta(months=4)
    s0_quater = s0_quater.strftime("%Y-%m-%d")

    sT_quater = datetime.strptime(sT, "%Y-%m-%d")
    sT_quater = sT_quater + relativedelta(months=4)
    sT_quater = sT_quater.strftime("%Y-%m-%d")

    query = (
        "SELECT "
        "c.gvkey, "
        "c.datadate, "
        "c.fyearq, "
        "c.fqtr, "
        "c.dlcq, "
        "c.dlttq, "
        "n.tic "
        "FROM "
        "comp.fundq AS c "
        "LEFT JOIN "
        "comp.names AS n "
        "ON "
        "c.gvkey = n.gvkey "
        "WHERE "
        "c.gvkey = '" + gvkey + "' "
        "AND c.datadate BETWEEN '" + s0_quater + "' AND '" + sT_quater + "' "
        "ORDER BY "
        "c.datadate;"
    )
    D = conn.raw_sql(query)
    D["dlcq"] = D["dlcq"] * 1e6
    D["dlttq"] = D["dlttq"] * 1e6


    D.index = pd.to_datetime(D.datadate)
    D = D.drop("datadate", axis = 1)

    return D[["dlcq", "dlttq"]]

