from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF


conf = SparkConf().setMaster("local").setAppName("WikipediaSearcher")
sc = SparkContext(conf = conf)

rawData = sc.textFile("C:/Users/User/Contacts/Desktop/wikipedia article searcher/subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))


documentNames = fields.map(lambda x: x[1])

hashingTermFrequency = HashingTF(100000)  
termFrequency = hashingTermFrequency.transform(documents)

termFrequency.cache()
idf = IDF(minDocFreq=2).fit(termFrequency)
tfidf = idf.transform(termFrequency)


singaporeTF = hashingTermFrequency.transform(["Singapore"])
singaporeHashValue = int(singaporeTF.indices[0])

singaporeRelevance = tfidf.map(lambda x: x[singaporeHashValue])


zippedResults = singaporeRelevance.zip(documentNames)


print("Best document for Singapore is:")
print(zippedResults.max())
