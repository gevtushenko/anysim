#include "gtest/gtest.h"
#include "core/config/configuration.h"

const static double epsilon = 1e-10;

TEST(configuration, leaf_nodes)
{
  configuration config;

  // Bool
  {
    const bool target_value = true;
    const auto fid = config.create_node ("frequency", target_value);
    const bool actual_value = config.get_node_value (fid);

    ASSERT_EQ (target_value, actual_value);
  }

  {
    const bool target_value = false;
    const auto fid = config.create_node ("frequency", target_value);
    const bool actual_value = config.get_node_value (fid);

    ASSERT_EQ (target_value, actual_value);
  }

  // Double
  {
    const double target_value = 42.0;
    const auto fid = config.create_node ("frequency", target_value);
    const double actual_value = config.get_node_value (fid);

    ASSERT_NEAR (target_value, actual_value, epsilon);
  }

  // Int
  {
    const int target_value = 10;
    const auto fid = config.create_node ("id", target_value);
    const int actual_value = config.get_node_value (fid);

    ASSERT_EQ (target_value, actual_value);
  }

  // String
  {
    const std::string target_value = "Source 1";
    const auto fid = config.create_node ("source_name", target_value);
    const std::string actual_value = config.get_node_value (fid);

    ASSERT_EQ (target_value, actual_value);
  }
}

