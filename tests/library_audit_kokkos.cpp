// Independent API-level regressions from the companion-library audit.
#include "mstar/mstar.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Integration_diagnostics, contact_present_at_entry_booked_once) {
    mstar::System s(0,3);
    s.constants.G=s.constants.c=1;
    s.options.PN_flags=0;
    s.options.gbs_relative_tolerance=1e-12;
    s.options.output_time_relative_tolerance=1e-12;
    s.options.star_star_softening=.1;
    s.options.max_step_count=10000;
    s.mass={1,1,1};
    s.pos={-.005,0,0,.005,0,0,1,0,0};
    s.vel.assign(9,0);
    s.collision_radius={.01,.01,0};
    const double entry_energy=s.energy();
    ASSERT_TRUE(s.run(1e-5));
    ASSERT_EQ(s.mergers.size(),1u);
    ASSERT_EQ(s.num_particles,2);
    double collision_work=0;
    for(const auto& m:s.mergers)collision_work+=m.dE_collision;
    const double actual=(s.energy()-entry_energy-collision_work)/entry_energy;
    EXPECT_NEAR(s.perf.relative_energy_error,actual,1e-10);
}

TEST(Integration_limits, does_not_exceed_requested_attempts) {
    mstar::System s(0,2);
    s.constants.G=s.constants.c=1;
    s.options.PN_flags=0;
    s.options.enable_bh_mergers=false;
    s.options.max_step_count=1;
    s.mass={.5,.5};
    s.pos={.95,0,0,-.95,0,0};
    const double v=.5*std::sqrt(.1/1.9);
    s.vel={0,v,0,0,-v,0};
    EXPECT_FALSE(s.run(100));
    EXPECT_EQ(s.perf.successful_steps+s.perf.failed_steps,1);
}
