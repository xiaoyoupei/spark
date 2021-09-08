package com.znv.spark.mllib

import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Demo09WordsSplit {
  def main (args: Array[String]): Unit = {
    //1、构建spark环境
    val spark: SparkSession = SparkSession
      .builder()
      .appName("person")
      .master("local[*]")
      .config("spark,sql.shuffle.partitions", "2")
      .getOrCreate()

    import spark.implicits._
    import org.apache.spark.sql.functions._

    //2、读取文本数据
    val data: DataFrame = spark
      .read
      .format("csv")
      .option("sep", "\t")
      .schema("label DOUBLE,text STRING")
      .load("data/train1.txt")

    //3、分词
    val wordsDS: Dataset[(Double, List[String])] = data
      .as[(Double, String)]
      .map(kv => {
        val label: Double = kv._1
        val text: String = kv._2

        val words: List[String] = Demo08IK.fit(text) //调用IK分词器进行分词
        (label, words)
      })
    //wordsDS.show(1000,false)

    //4、去除脏数据
    val filterDS: Dataset[(Double, List[String])] = wordsDS.filter(_._2.length > 2)


    //5、将集合中每一个词语使用空格拼接
    val linesDF: DataFrame = filterDS
      .map(kv => {
        (kv._1, kv._2.mkString(" "))
      }).toDF("label", "text")

    //6、使用官方提供的英文分词器，转为空格拼接也是为了方便使用英文分词器
    val tokenizer: Tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val wordsData: DataFrame = tokenizer.transform(linesDF)

    //7、计算TF(词频)和IDF(逆文本频率)
    val hashingTF: HashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures")
    val featurizedData: DataFrame = hashingTF.transform(wordsData)

    val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features") //IDF
    val idfModel: IDFModel = idf.fit(featurizedData) //训练IDF模型
    val rescaledData: DataFrame = idfModel.transform(featurizedData) //

    //8、将结果拆分训练集和测试集
    val array: Array[Dataset[Row]] = rescaledData.randomSplit(Array(0.7, 0.3))
    val train: Dataset[Row] = array(0)
    val test: Dataset[Row] = array(1)


    //9、采用贝叶斯分类做问文本分类
    val model: NaiveBayesModel = new NaiveBayes().fit(train)
    val dataFrame: DataFrame = model.transform(test)

    //10、计算准确率
    dataFrame
      .select(sum(when($"label" === $"prediction", 1).otherwise(0)) / count($"label") as "rate")
      .show()

    //11、保存IDF模型和贝叶斯模型
    idfModel.write.overwrite().save("data/idfModel")
    model.write.overwrite().save("data/naiveBayes")

  }
}
