/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tree.impurity

import org.apache.spark.annotation.{DeveloperApi, Experimental, Since}

/**
 * :: Experimental ::
 * Class for calculating the
 * [[http://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity Gini impurity]]
 * during multiclass classification.
 */
@Since("1.0.0")
@Experimental
object Gini extends Impurity {

  /**
   * :: DeveloperApi ::
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @Since("1.1.0")
  @DeveloperApi
  override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0
    }
    val numClasses = counts.length
    var impurity = 1.0
    var classIndex = 0
    while (classIndex < numClasses) {
      val freq = counts(classIndex) / totalCount
      impurity -= freq * freq
      classIndex += 1
    }
    impurity
  }

  /**
   * :: DeveloperApi ::
   * variance calculation
   * @param count number of instances
   * @param sum sum of labels
   * @param sumSquares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  @Since("1.0.0")
  @DeveloperApi
  override def calculate(count: Double, sum: Double, sumSquares: Double): Double =
    throw new UnsupportedOperationException("Gini.calculate")

  def calculate(allStats: Array[Double], offset: Int, statsSize: Int): Double = {
    var index = offset
    val endIndex = offset + statsSize
    var totalCount = 0.0
    while (index < endIndex) {
      totalCount += allStats(index)
      index += 1
    }

    if (totalCount == 0) {
      return 0
    }

    var impurity = 1.0
    index = offset
    while (index < endIndex) {
      val freq = allStats(index) / totalCount
      impurity -= freq * freq
      index += 1
    }
    impurity
  }

  /**
   * Get this impurity instance.
   * This is useful for passing impurity parameters to a Strategy in Java.
   */
  @Since("1.1.0")
  def instance: this.type = this

}

/**
 * Class for updating views of a vector of sufficient statistics,
 * in order to compute impurity from a sample.
 * Note: Instances of this class do not hold the data; they operate on views of the data.
 * @param numClasses  Number of classes for label.
 */
private[spark] class GiniAggregator(numClasses: Int)
  extends ImpurityAggregator(numClasses) with Serializable {

  /**
   * Update stats for one (node, feature, bin) with the given label.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def update(allStats: Array[Double], offset: Int, label: Double, instanceWeight: Double): Unit = {
    if (label >= statsSize) {
      throw new IllegalArgumentException(s"GiniAggregator given label $label" +
        s" but requires label < numClasses (= $statsSize).")
    }
    if (label < 0) {
      throw new IllegalArgumentException(s"GiniAggregator given label $label" +
        s"but requires label is non-negative.")
    }
    allStats(offset + label.toInt) += instanceWeight
  }

  /**
   * Get an [[ImpurityCalculator]] for a (node, feature, bin).
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def getCalculator(allStats: Array[Double], offset: Int): GiniCalculator = {
    new GiniCalculator(
      allStats.view(offset, offset + statsSize).toArray, allStats, offset, statsSize)
  }
 }

/**
 * Stores statistics for one (node, feature, bin) for calculating impurity.
 * Unlike [[GiniAggregator]], this class stores its own data and is for a specific
 * (node, feature, bin).
 * @param stats  Array of sufficient statistics for a (node, feature, bin).
 */
private[spark] class GiniCalculator(
    stats: Array[Double],
    var allStats: Array[Double],
    offset: Int, statsSize: Int) extends ImpurityCalculator(stats) {

  /**
   * Make a deep copy of this [[ImpurityCalculator]].
   */
  def copy: GiniCalculator = new GiniCalculator(stats.clone(), allStats.clone(), offset, statsSize)

  /**
   * Calculate the impurity from the stored sufficient statistics.
   */
  def calculate(): Double = Gini.calculate(allStats, offset, statsSize)

  /**
   * Number of data points accounted for in the sufficient statistics.
   */
  def count: Long = {
    var index = offset
    val endIndex = offset + statsSize
    var totalCount = 0.0
    while (index < endIndex) {
      totalCount += allStats(index)
      index += 1
    }
    totalCount.toLong
  }

  /**
   * Prediction which should be made based on the sufficient statistics.
   */
  def predict: Double = if (count == 0) {
    0
  } else {
    indexOfLargestArrayElement(allStats)
  }

  override def indexOfLargestArrayElement(stats: Array[Double]): Int = {
    var largest = -1
    var largestValue = Double.MinValue
    var index = 0
    while (index < statsSize) {
      if (stats(index + offset) > largestValue) {
        largest = index
        largestValue = stats(index + offset)
      }
      index += 1
    }
    largest
  }

  override def add(other: ImpurityCalculator): ImpurityCalculator = {
    require(stats.length == other.stats.length,
      s"Two ImpurityCalculator instances cannot be added with different counts sizes." +
        s"  Sizes are ${stats.length} and ${other.stats.length}.")
    var i = 0
    val len = other.stats.length
    val newAllStats = allStats.clone()
    while (i < len) {
      stats(i) += other.stats(i)
      newAllStats(i + offset) += other.stats(i)
      i += 1
    }
    this.allStats = newAllStats
    this
  }

  /**
   * Subtract the stats from another calculator from this one, modifying and returning this
   * calculator.
   */
  override def subtract(other: ImpurityCalculator): ImpurityCalculator = {
    require(stats.length == other.stats.length,
      s"Two ImpurityCalculator instances cannot be subtracted with different counts sizes." +
      s"  Sizes are ${stats.length} and ${other.stats.length}.")
    var i = 0
    val len = other.stats.length
    val newAllStats = allStats.clone()
    while (i < len) {
      stats(i) -= other.stats(i)
      newAllStats(i + offset) -= other.stats(i)
      i += 1
    }
    this.allStats = newAllStats
    this
  }

  /**
   * Probability of the label given by [[predict]].
   */
  override def prob(label: Double): Double = {
    val lbl = label.toInt
    require(lbl < statsSize,
      s"GiniCalculator.prob given invalid label: $lbl (should be < ${statsSize}")
    require(lbl >= 0, "GiniImpurity does not support negative labels")
    val cnt = count
    if (cnt == 0) {
      0
    } else {
      allStats(lbl + offset) / cnt
    }
  }

  override def toString: String = s"GiniCalculator(stats = [${stats.mkString(", ")}])"

}
