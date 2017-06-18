# simple test

N = 20000
sum = 0
alpha0 = 0.5
for i in xrange(1, N+1):
    product = 1
    for j in xrange(i+1, N+1):
        product *= (j-alpha0)/j
    #print 'product', product, ' current ', 1.0/i

    sum += alpha0/i*product

print 'sum: ',  sum