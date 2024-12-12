#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include "parse_time.h"

TEST(ParseTimeTest, ValidFractionalInputs) {
    EXPECT_DOUBLE_EQ(parseTime("0.5h"), 0.5 / 24.0);
    EXPECT_DOUBLE_EQ(parseTime("0.75d"), 0.75);
    EXPECT_DOUBLE_EQ(parseTime("0.1y"), 36.525);
}

TEST(ParseTimeTest, InvalidUnits) {
    EXPECT_THROW(parseTime("1m"), std::invalid_argument);
    EXPECT_THROW(parseTime("2s"), std::invalid_argument);
    EXPECT_THROW(parseTime("10"), std::invalid_argument);
}

TEST(ParseTimeTest, MissingNumber) {
    EXPECT_THROW(parseTime("h"), std::invalid_argument);
    EXPECT_THROW(parseTime("d"), std::invalid_argument);
}

TEST(ParseTimeTest, NegativeTime) {
    EXPECT_DOUBLE_EQ(parseTime("-1h"), -1.0 / 24.0);
    EXPECT_DOUBLE_EQ(parseTime("-2d"), -2.0);
    EXPECT_DOUBLE_EQ(parseTime("-0.5y"), -182.625);
}

