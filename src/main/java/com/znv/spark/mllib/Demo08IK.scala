package com.znv.spark.mllib

import java.io.StringReader

import org.wltea.analyzer.core.{IKSegmenter, Lexeme}

import scala.collection.mutable.ListBuffer

object Demo08IK {
  def main (args: Array[String]): Unit = {
    val str = "别人笑我太疯癫，我笑他人看不穿；不见五陵豪杰墓，无花无酒锄作田。"

    val words: List[String] = fit(str)
    println(words)

  }

  def fit (text: String): List[String] = {
    val listBuffer: ListBuffer[String] = new ListBuffer[String]

    val sr: StringReader = new StringReader(text)

    val ik: IKSegmenter = new IKSegmenter(sr, true)

    var word: Lexeme = ik.next()

    while (word != null) {
      listBuffer += word.getLexemeText

      word = ik.next()
    }

    listBuffer.toList

  }
}
