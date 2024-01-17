def maxProfit(prices):
    max_prof = 0
    close_iter = len(prices) - 1
    for idx, open_val in enumerate(prices):
        temp_close_iter = idx + 1 
        while temp_close_iter <= close_iter:
            xx = (prices[temp_close_iter] - open_val)
            if ( xx > max_prof ):
                max_prof = xx
            temp_close_iter += 1
    return max_prof
            
    
print(maxProfit([1]))