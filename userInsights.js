function generateNiftyPerformanceSummary(performanceData) {
    const { index, currentValue, previousClose, change, percentageChange, high, low } = performanceData;
  
    // Generate the summary
    let summary = `The ${index} is currently trading at ${currentValue}. `;
    summary += `This is an increase of ${change} points from the previous close of ${previousClose}, `;
    summary += `which represents a ${percentageChange.toFixed(2)}% change. `;
    summary += `The day's high so far has been ${high} and the low has been ${low}.`;
  
    return summary;
  }
  
  // Sample Nifty performance data
  const niftyPerformance = {
    index: 'Nifty 50',
    currentValue: 19000,
    previousClose: 18800,
    change: 200,
    percentageChange: 1.06,
    high: 19100,
    low: 18750,
  };
  
  // Generate and log the summary
  const summary = generateNiftyPerformanceSummary(niftyPerformance);
  console.log(summary);
  