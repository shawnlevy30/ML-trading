import json
import pandas_ta as ta
import talib

def list_pandas_ta_indicators():
    # retrieves all available indicators from pandas_ta and returns them as a list
    categories = ta.Category
    indicators = []

    for category_name, indicator_list in categories.items():
        indicators.extend(indicator_list)
    
    return indicators

def list_talib_indicators():

    indicators = talib.get_functions()
    return indicators

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent = 4)
    print(f"Saved {len(data)} to {filename}")

def main():
    print("=== indicator listing script started ===\n")

    print("listing pandas_ta indicators...")
    pandas_ta_indicators = list_pandas_ta_indicators()
    save_to_json(
        pandas_ta_indicators, 
        '/Users/shawnlevy/Desktop/BootCamp/ML/results/pandas_ta_indicators.json'
    )

    print("\nlisting talib indicators...")
    talib_indicators = list_talib_indicators()
    save_to_json(
        talib_indicators, 
        '/Users/shawnlevy/Desktop/BootCamp/ML/results/talib_indicators.json'
    )

    print ("\n Indicator listing complete")

if __name__ == "__main__":
    main()

