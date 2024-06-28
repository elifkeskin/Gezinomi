##################################################################################
# POTENTIAL CUSTOMER GAININGS CALCULATION WITH RULE-BASED CLASSIFICATION
##################################################################################

###################################################
# DATASET GENERAL INFORMATION
###################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_excel("gezinomi_tanıtım/miuul_gezinomi.xlsx")
df.head()
df.shape   # (59164, 9)
df.info()
# #   Column              Non-Null Count  Dtype
# ---  ------              --------------  -----
#  0   SaleId              59164 non-null  int64
#  1   SaleDate            59164 non-null  datetime64[ns]
#  2   CheckInDate         59164 non-null  datetime64[ns]
#  3   Price               59151 non-null  float64
#  4   ConceptName         59164 non-null  object
#  5   SaleCityName        59164 non-null  object
#  6   CInDay              59164 non-null  object
#  7   SaleCheckInDayDiff  59164 non-null  int64
#  8   Seasons             59164 non-null  object
# dtypes: datetime64[ns](2), float64(1), int64(2), object(4)
# memory usage: 4.1+ MB

"""
Değişkenler:
SaleId: Satış id
SaleDate : Satış Tarihi
Price: Satış için ödenen fiyat
ConceptName:Otel konsept bilgisi
SaleCityName: Otelin bulunduğu şehir bilgisi
CheckInDate: Müşterinin otele giriş tarihi
CInDay:Müşterinin otele giriş günü
SaleCheckInDayDiff: Check in ile giriş tarihi gün farkı
Season:Otele giriş tarihindeki sezon bilgisi
"""

#########################################
# EDA (Exploratory Data Analysis)
#########################################

# How many unique cities are there? What are their frequencies?
df['SaleCityName'].nunique()  # 6
df['SaleCityName'].value_counts()
# SaleCityName
# Antalya    31649
# Muğla      10662
# Aydın      10646
# Diğer       3245
# İzmir       2507
# Girne        455
# Name: count, dtype: int64

# How many unique concepts are there?
df['ConceptName'].nunique()  # 3

# How many sales were made from which concept?
df['ConceptName'].value_counts()
# ConceptName
# Herşey Dahil      53186
# Yarım Pansiyon     3559
# Oda + Kahvaltı     2419
# Name: count, dtype: int64

# How much was earned from sales in total by city?
df.groupby('SaleCityName').agg({'Price':'sum'})

#                   Price
# SaleCityName
# Antalya      2041911.10
# Aydın         573296.01
# Diğer         154572.29
# Girne          27065.03
# Muğla         665842.21
# İzmir         165934.83

# How much has been earned according to concept types?
df.groupby('ConceptName').agg({'Price':'sum'})

#                     Price
# ConceptName
# Herşey Dahil   3332910.77
# Oda + Kahvaltı  121308.35
# Yarım Pansiyon  174402.35

# What are the price averages by city?
df.groupby('SaleCityName').agg({'Price':'mean'})
#              Price
# SaleCityName
# Antalya       64.52
# Aydın         53.86
# Diğer         47.71
# Girne         59.48
# Muğla         62.46
# İzmir         66.27

# What is the average price according to concepts?
df.groupby('ConceptName').agg({'Price':'mean'})

#                 Price
# ConceptName
# Herşey Dahil    62.67
# Oda + Kahvaltı  50.25
# Yarım Pansiyon  49.03

# What are the price averages in city-concept breakdown?
df.groupby(['SaleCityName', 'ConceptName']).agg({'Price':'mean'})

#                             Price
# SaleCityName ConceptName
# Antalya      Herşey Dahil    64.52
#              Oda + Kahvaltı  63.50
#              Yarım Pansiyon  67.19
# Aydın        Herşey Dahil    54.00
#              Oda + Kahvaltı  34.46
#              Yarım Pansiyon  30.02
# Diğer        Herşey Dahil    84.77
#              Oda + Kahvaltı  37.60
#              Yarım Pansiyon  42.11
# Girne        Herşey Dahil    97.68
#              Oda + Kahvaltı  39.78
#              Yarım Pansiyon  53.25
# Muğla        Herşey Dahil    63.02
#              Oda + Kahvaltı  59.04
#              Yarım Pansiyon  45.12
# İzmir        Herşey Dahil    74.70
#              Oda + Kahvaltı  41.32
#              Yarım Pansiyon  59.61

#########################################
# Feature Engineering
#########################################

# Converting the SaleCheckInDayDiff variable to a categorical variable

bins = [-1, 7, 30, 90, df['SaleCheckInDayDiff'].max()]
labels = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]

# The cut function allows us to convert our numerical variables into categorical variables.
# The cut function asks us what to divide and at what intervals.
df["EB_Score"] = pd.cut(df["SaleCheckInDayDiff"], bins=bins, labels=labels)
df.head()

# Average price and number of transactions made in the City-Concept-EB_Score breakdown:
df.groupby(['SaleCityName', 'ConceptName', 'EB_Score']).agg({'Price':['mean', 'count']})

# Average price and number of transactions made according to City-Concept-Season breakdown:
df.groupby(['SaleCityName', 'ConceptName', 'Seasons']).agg({'Price':['mean', 'count']})

# Average price and number of transactions made in the City-Concept-CInDay breakdown:
df.groupby(['SaleCityName', 'ConceptName', 'CInDay']).agg({'Price':['mean', 'count']})


# Let's sort the output of the City-Concept-Season breakdown by price and save it as agg_df.

agg_df = df.groupby(['SaleCityName', 'ConceptName', 'Seasons']).agg({'Price':'mean'}).sort_values('Price', ascending=False)
agg_df.head()
#                                    Price
# SaleCityName ConceptName    Seasons
# Girne        Herşey Dahil   High    103.94
#                             Low      90.94
# İzmir        Yarım Pansiyon High     87.66
# Diğer        Herşey Dahil   Low      87.31
#                             High     83.79

agg_df.reset_index(inplace=True)
agg_df.head()
# SaleCityName     ConceptName Seasons  Price
# 0        Girne    Herşey Dahil    High 103.94
# 1        Girne    Herşey Dahil     Low  90.94
# 2        İzmir  Yarım Pansiyon    High  87.66
# 3        Diğer    Herşey Dahil     Low  87.31
# 4        Diğer    Herşey Dahil    High  83.79

agg_df['sales_level_based'] = agg_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '-'.join(x).upper(), axis=1)
agg_df.head()

# Let's divide it into 4 segments according to PRICE.
# Let's add the segments to the agg_df file with the name "SEGMENT".
#Let's group the segments by price and get their mean, max and sum.

agg_df['SEGMENT'] = pd.qcut(agg_df["Price"], 4, labels=['D', 'C', 'B', 'A'])
agg_df.head(30)
agg_df.groupby('SEGMENT').agg({'Price':['mean', 'max', 'sum']})

# Price
#          mean    max    sum
# SEGMENT
# D       33.37  39.48 300.30
# C       44.89  54.14 403.99
# B       60.27  64.92 542.47
# A       82.47 103.94 742.21

# Let's sort the last variable according to the price variable and in which segment is "ANTALYA_HERŞEY DAHIL_HIGH"?
# and how much price is expected?
agg_df.sort_values('Price', ascending=False)

new_user = "ANTALYA-HERŞEY DAHIL-HIGH"
agg_df[agg_df["sales_level_based"] == new_user]

# SaleCityName   ConceptName Seasons  Price          sales_level_based  SEGMENT
# 9      Antalya  Herşey Dahil    High  64.92  ANTALYA-HERŞEY DAHIL-HIGH    B

# In which segment will a holidaymaker who goes to a half-board hotel in Kyrenia in low season be included?
new_user = "GIRNE-YARIM PANSIYON-LOW"
agg_df[agg_df["sales_level_based"] == new_user]


# SaleCityName     ConceptName Seasons  Price         sales_level_based     SEGMENT
# 19        Girne  Yarım Pansiyon     Low  48.58  GIRNE-YARIM PANSIYON-LOW       C