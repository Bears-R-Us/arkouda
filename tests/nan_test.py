from context import arkouda as ak
import numpy as np
import pandas as pd
from base_test import ArkoudaTest

class NanTest(ArkoudaTest):

    def setUp(self):
        ArkoudaTest.setUp(self)
        SIZE = 5000
        self.a = ak.randint(0, 2*SIZE, SIZE, ak.float64)
        self.group = ak.GroupBy(ak.randint(0,1,SIZE))
        self.a[0] = np.nan
        
    def testGroupMean(self): 
        group_mean = (self.group).mean(self.a,True)

        # calculate real mean excluding nan
        sum = 0
        for i in range(1,self.a.size):
            sum += self.a[i]
        mean = sum/(self.a.size-1)

        print(mean)
        print(group_mean[1][0])
        
        self.assertTrue(abs(group_mean[1][0] -  mean) <= self.a.size * .0001)

    def testGroupMin(self):
        group_min = (self.group).min(self.a,True)

        minVal = self.a.size*3 #bigger than possible value
        # calculate the min
        for i in range(1,self.a.size):
            if self.a[i] < minVal:
                minVal = self.a[i]

        print(minVal)
        print(group_min[1][0])

        self.assertTrue(group_min[1][0] == minVal)

    def testGroupSum(self):
        group_sum = (self.group).sum(self.a,True)

        # calculate the sum exlcuding nan
        sum = 0
        for i in range(1,self.a.size):
            sum += self.a[i]

        print(sum)
        print(group_sum[1][0])

        self.assertTrue(abs(group_sum[1][0] -  sum) <= self.a.size * .0001)
        
    def testGroupMax(self):
        group_max = (self.group).max(self.a,True)

        maxVal = -1 # smaller than possible value
        # calculate the min
        for i in range(1,self.a.size):
            if self.a[i] > maxVal:
                maxVal = self.a[i]

        print(maxVal)
        print(group_max[1][0])
                
        self.assertTrue(group_max[1][0] == maxVal)
