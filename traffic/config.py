from pathlib import Path

numApps = 3  # number of categories of Apps with different delay budgets
packetsPerUE = 50
packetSize = 5e3  # bits per packet
fileSize = packetsPerUE * packetSize
delayBudgets = [.05, .15, .30]  # independent budget for each newly generated packet
appNames = ['Delay Stringent', 'Delay Sensitive', 'Delay Tolerant']
dpiSampleRates = [1/5] * 3  # profiles are already expanded for DPI sampling
profilesPath = Path(__file__).parent.parent/'data/site_traffic_profiles.csv'
