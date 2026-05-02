import re

# Paste your raw output text here
raw_data = """
Cost Value: 157477.985121, time: 9.2521 s, iterations: 200
Cost Value: 124681.289856, time: 9.2835 s, iterations: 200
Cost Value: 125670.509717, time: 9.2886 s, iterations: 200
Cost Value: 134284.168975, time: 9.4151 s, iterations: 200
Cost Value: 130747.064164, time: 9.0934 s, iterations: 200
Cost Value: 147698.18229, time: 9.1452 s, iterations: 200
Cost Value: 144711.652588, time: 9.1886 s, iterations: 200
Cost Value: 125226.445023, time: 9.2303 s, iterations: 200
Cost Value: 139735.960062, time: 9.2893 s, iterations: 200
Cost Value: 131114.904942, time: 9.2478 s, iterations: 200
"""

def summarize_metrics(text):
    # Extract numerical values using Regex
    costs = [float(x) for x in re.findall(r"Cost Value: ([\d.]+)", text)]
    times = [float(x) for x in re.findall(r"time: ([\d.]+) s", text)]
    iters = [int(x) for x in re.findall(r"iterations: (\d+)", text)]

    if not costs:
        return "No data found."

    # Calculate summary statistics
    summary = {
        "Count": len(costs),
        "Avg Cost": sum(costs) / len(costs),
        "Avg Time": sum(times) / len(times),
        "Avg Iterations": sum(iters) / len(iters),
        "Min Cost": min(costs),
        "Max Cost": max(costs)
    }
    
    return summary

# Run and print results
results = summarize_metrics(raw_data)
print("--- Data Summary ---")
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:,.4f}")
    else:
        print(f"{key}: {value}")

