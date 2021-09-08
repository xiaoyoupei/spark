package com.znv.spark.mllib

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

object Demo04Kmeans {
  def main (args: Array[String]): Unit = {
    //1、构建spark环境
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Kmeans")
      .master("local")
      .config("spark,sql.shuffle.partitions", "2")
      .getOrCreate()

    import spark.implicits._

    //2、读取数据
    val data: DataFrame = spark
      .read
      .format("csv")
      .schema("x DOUBLE,y DOUBLE")
      .load("data/kmeans.txt")

    //3、将每行数据拼接成数组
    val ds: Dataset[(Double, Double)] = data.as[(Double, Double)] //每行转为元组
    val vectorDF: DataFrame = ds
      .map(kv => Array(kv._1, kv._2))
      .toDF("features") //必须要加这个列名，不然无法获取数据会报错

    //4、构建Kmeans算法
    val Kmeans: KMeans = new KMeans()
      .setK(3) //k的数量，聚类中心的数量

    //5、迭代计算训练模型
    val model: KMeansModel = Kmeans.fit(vectorDF)

    //6、计算结果
    val result: DataFrame = model.transform(vectorDF)

    result.show(100000) //查看结果

  }
}
