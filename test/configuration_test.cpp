#include "gtest/gtest.h"
#include "core/config/configuration.h"

const static double epsilon = 1e-10;

TEST(configuration, base)
{
  configuration config;
  const double target_value = 42.0;
  const auto fid = config.create_node ("frequency", target_value);
  const double actual_value = config.get_node_value (fid);

  ASSERT_NEAR (target_value, actual_value, epsilon);
}

