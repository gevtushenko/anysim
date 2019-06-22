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

    ASSERT_EQ (config.get_node_type (fid), bool_type);
    ASSERT_EQ (target_value, actual_value);
  }

  {
    const bool target_value = false;
    const auto fid = config.create_node ("frequency", target_value);
    const bool actual_value = config.get_node_value (fid);

    ASSERT_EQ (config.get_node_type (fid), bool_type);
    ASSERT_EQ (target_value, actual_value);
  }

  // Double
  {
    const double target_value = 42.0;
    const auto fid = config.create_node ("frequency", target_value);
    const double actual_value = config.get_node_value (fid);

    ASSERT_EQ (config.get_node_type (fid), double_type);
    ASSERT_NEAR (target_value, actual_value, epsilon);
  }

  // Int
  {
    const int target_value = 10;
    const auto fid = config.create_node ("id", target_value);
    const int actual_value = config.get_node_value (fid);

    ASSERT_EQ (config.get_node_type (fid), int_type);
    ASSERT_EQ (target_value, actual_value);
  }

  // String
  {
    const std::string target_value = "Source 1";
    const auto fid = config.create_node ("source_name", target_value);
    const std::string actual_value = config.get_node_value (fid);

    ASSERT_EQ (config.get_node_type (fid), string_type);
    ASSERT_EQ (target_value, actual_value);
  }
}

TEST(configuration, group_nodes)
{
  configuration config;
  const auto sources = config.create_array ("sources");
  const auto source_1 = config.create_group ("source_1");
  const auto source_2 = config.create_group ("source_2");

  {
    const auto nid = config.create_node ("name", std::string ("Source 1"));
    const auto fid = config.create_node ("frequency", 1.E+10);
    const auto xid = config.create_node ("x", 1);
    const auto yid = config.create_node ("y", 1);

    config.add_child (source_1, nid);
    config.add_child (source_1, fid);
    config.add_child (source_1, xid);
    config.add_child (source_1, yid);
  }

  {
    const auto nid = config.create_node ("name", std::string ("Source 2"));
    const auto fid = config.create_node ("frequency", 2.E+10);
    const auto xid = config.create_node ("x", 2);
    const auto yid = config.create_node ("y", 3);

    config.add_child (source_2, nid);
    config.add_child (source_2, fid);
    config.add_child (source_2, xid);
    config.add_child (source_2, yid);
  }

  config.add_children (sources, { source_1, source_2 });
}
