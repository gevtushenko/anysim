#include "gtest/gtest.h"
#include "core/config/configuration.h"

const static double epsilon = 1e-10;

TEST(configuration, base)
{
  configuration config;
  auto root = config.get_root ();
  ASSERT_EQ (root.name, "root");
  ASSERT_EQ (config.get_version (), 1);

  const configuration const_config;
  auto const_root = config.get_root ();
  ASSERT_EQ (const_config.get_version (), 0);

  root.append_node ("v_1", 1).append_node ("v_2", 2.0);
  ASSERT_EQ (root.child (0).name, "v_1");
  ASSERT_EQ (std::get <int> (root.child (0).value), 1);
  ASSERT_EQ (root.child (1).name, "v_2");
  ASSERT_NEAR (std::get <double> (root.child (1).value), 2.0, epsilon);
  ASSERT_ANY_THROW (std::get <int> (root.child (1).value));
}

TEST(configuration, groups)
{
  configuration config;
  auto sources = config.get_root ().append_group ("sources");
  sources.append_group ("s_1").append_group ("s_2").append_group ("s_3");
  sources.child (0).append_node ("frequency", 1.E+9).append_node ("x", 0.0).append_node ("y", 0.0);
  sources.child (1).append_node ("frequency", 1.E+9).append_node ("x", 1.0).append_node ("y", 1.0);
  sources.child (2).append_node ("frequency", 1.E+9).append_node ("x", 2.0).append_node ("y", 2.0);

  auto sources_group = config.get_root ().group (0);
  for (unsigned int gid = 0; gid < sources_group.size (); gid++)
  {
    ASSERT_EQ (sources_group[gid].name, "s_" + std::to_string (gid));
    ASSERT_NEAR (std::get<double> (sources_group[gid].child (1).value), double (gid), epsilon);
  }
}
