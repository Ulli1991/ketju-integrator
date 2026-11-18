// Controlled fixed-kappa SDAR experiment. No production/GIZMO configuration is changed.
#include "mstar/mstar.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using V3=std::array<double,3>;
static double dot(V3 a,V3 b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V3 cross(V3 a,V3 b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static double norm(V3 a){return std::sqrt(dot(a,a));}
static V3 minus(V3 a,V3 b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static V3 val(const std::vector<double>& a,int i){return {a[3*i],a[3*i+1],a[3*i+2]};}
static double binary_energy(const mstar::System& s){
    V3 r=minus(val(s.pos,1),val(s.pos,0)),v=minus(val(s.vel,1),val(s.vel,0));
    double mu=s.mass[0]*s.mass[1]/(s.mass[0]+s.mass[1]);
    return .5*mu*dot(v,v)-s.constants.G*s.mass[0]*s.mass[1]/norm(r);
}
static double effective_energy(const mstar::System& s,double k){return s.energy()+(1/k-1)*binary_energy(s);}
static V3 angular(const mstar::System& s){
    V3 l={0,0,0}; for(int i=0;i<s.num_particles;i++) {auto q=cross(val(s.pos,i),val(s.vel,i)); for(int d=0;d<3;d++) l[d]+=s.mass[i]*q[d];} return l;
}
struct State {
    std::vector<double> m,x,v;
    double t=0,seconds=0,a=0,e=0,energy_error=0,physical_energy_change=0,L_error=0,applied=1;
    V3 h{},outer{};
    long accepted=0,rejected=0;
    bool ok=false, completed=false;
};
// 0: isolated, 1: weak triple, 2: prompt encounter, 3: incoming perturber.
static double duration_for(int kind) {
    if (kind == 2) return 5;
    if (kind == 3) return 18;
    if (kind == 4) return 180;
    return 128 * std::acos(-1.);
}
static bool is_encounter(int kind) { return kind >= 2; }

static void setup(mstar::System& s,int kind){
    s.constants.G=s.constants.c=1; s.options.PN_flags=0; s.options.enable_bh_mergers=false;
    s.options.gbs_relative_tolerance=1e-10; s.options.output_time_relative_tolerance=1e-12;
    s.options.output_time_absolute_tolerance=1e-15; s.options.max_step_count=1000000;
    const double e=.4,ra=1+e,vr=std::sqrt((1-e)/(1+e));
    s.mass[0]=s.mass[1]=.5; s.pos.assign(3*s.num_particles,0); s.vel.assign(3*s.num_particles,0);
    s.pos[0]=ra/2; s.pos[3]=-ra/2; s.vel[1]=vr/2; s.vel[4]=-vr/2;
    if(s.num_particles==3){
        double R=kind==2?6:(kind>=3?30:50); s.mass[2]=.5;
        V3 vo=is_encounter(kind)?V3{-2,1.8/R,0}:V3{0,.5*std::sqrt(1.5/R),std::sqrt(.75)*std::sqrt(1.5/R)};
        if(kind==4)vo={0,std::sqrt(1.5*(2./30-1./16)),0};
        for(int i=0;i<2;i++){s.pos[3*i]-=R/3; for(int d=0;d<3;d++)s.vel[3*i+d]-=vo[d]/3;}
        s.pos[6]=2*R/3; for(int d=0;d<3;d++)s.vel[6+d]=2*vo[d]/3;
    }
}
static State run(int kind,double k,bool enabled=true,int fallback=0){
    const int n=kind==0?2:3;
    mstar::System s(fallback==1?2:0,fallback==1?n-2:n); setup(s,kind);
    s.options.experimental_slowdown=enabled; s.options.slowdown_factor=k;
    if(fallback==2)s.options.star_star_softening=.01;
    if(fallback==3)s.collision_radius.assign(n,0);
    if(fallback==4){s.options.enable_external_tide=true;s.options.external_tidal_tensor[0]=1e-5;}
    if(fallback==5){s.options.extra.vdep=true;s.options.extra.D[0]=s.options.extra.D[4]=s.options.extra.D[8]=-1e-4;}
    if(fallback==6)s.perturbers={.5,100,0,0,0,0,0,0,0,0};
    double duration=fallback?.1:duration_for(kind);
    double effective0=effective_energy(s,k),physical0=s.energy(); V3 L0=angular(s);
    auto start=std::chrono::steady_clock::now(); bool ok=s.run(duration);
    State q; q.ok=ok; q.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    q.t=s.time; q.completed=ok&&std::fabs(q.t-duration)<=std::max(1e-15,1e-12*duration); q.m=s.mass;q.x=s.pos;q.v=s.vel; q.applied=s.perf.applied_slowdown_factor;
    q.accepted=s.perf.successful_steps;q.rejected=s.perf.failed_steps;
    V3 r=minus(val(s.pos,1),val(s.pos,0)),v=minus(val(s.vel,1),val(s.vel,0));
    q.h=cross(r,v); double gm=s.constants.G*(s.mass[0]+s.mass[1]),rlen=norm(r);
    q.a=1/(2/rlen-dot(v,v)/gm); auto ev=cross(v,q.h); for(int d=0;d<3;d++)ev[d]=ev[d]/gm-r[d]/rlen; q.e=norm(ev);
    if(n==3){for(int d=0;d<3;d++)q.outer[d]=s.pos[6+d]-(s.mass[0]*s.pos[d]+s.mass[1]*s.pos[3+d])/(s.mass[0]+s.mass[1]);}
    q.energy_error=q.applied==1 ? std::fabs((s.energy()-physical0)/physical0) : std::fabs((effective_energy(s,k)-effective0)/effective0);
    q.physical_energy_change=(s.energy()-physical0)/std::fabs(physical0);
    q.L_error=norm(minus(angular(s),L0))/norm(L0);
    return q;
}
static bool bits(const State&a,const State&b){return a.m==b.m&&a.x.size()==b.x.size()&&a.v.size()==b.v.size()&&std::memcmp(a.x.data(),b.x.data(),a.x.size()*sizeof(double))==0&&std::memcmp(a.v.data(),b.v.data(),a.v.size()*sizeof(double))==0&&a.t==b.t;}
static double median(std::vector<double> a){std::sort(a.begin(),a.end());return a[a.size()/2];}
int main(int argc,char**argv){
    mstar::ScopeGuard guard(argc,argv); int failures=0,hard_failures=0,speed_failures=0;
    for(int kind:{0,1,2,3,4}){
        State ref=run(kind,1,false),eq=run(kind,1,true);
        bool same=ref.completed&&eq.completed&&bits(ref,eq); std::printf("EQUIVALENCE case=%d bitwise=%d\n",kind,same);if(!same)failures++;
        std::vector<double> base_times;
        for(int rep=0;rep<3;rep++){auto b=run(kind,1,false);base_times.push_back(b.seconds);if(!b.completed)failures++;std::printf("BASE case=%d rep=%d seconds=%.9g steps=%ld rejected=%ld\n",kind,rep,b.seconds,b.accepted,b.rejected);}
        for(double k:{4.,16.}){
            std::vector<double> times; bool pass=true, completed=true;
            for(int rep=0;rep<3;rep++){
                State q=run(kind,k);times.push_back(q.seconds);completed=completed&&q.completed;
                std::printf("COMPLETION case=%d k=%.0f rep=%d ok=%d completed=%d actual_time=%.17g requested_time=%.17g\n",kind,k,rep,q.ok,q.completed,q.t,duration_for(kind));
                double da=std::fabs((q.a-ref.a)/ref.a),de=std::fabs(q.e-ref.e),dh=norm(minus(q.h,ref.h))/norm(ref.h),dx=kind?norm(minus(q.outer,ref.outer))/norm(ref.outer):0;
                bool good=ref.completed&&q.completed&&q.applied==k&&std::isfinite(q.energy_error)&&da<=1e-3&&de<=1e-3&&dh<=1e-3&&dx<=1e-3&&q.energy_error<=1e-7&&q.L_error<=1e-7;
                if(is_encounter(kind)) {
                    good=ref.completed&&q.completed&&q.applied==1&&bits(ref,q)
                         &&q.accepted==ref.accepted&&q.rejected==ref.rejected;
                }
                double phase_error=0;
                if(kind==0){
                    double mean=std::remainder(std::acos(-1.)+q.t/k,2*std::acos(-1.)),E=mean;
                    for(int it=0;it<100;it++){double d=(E-.4*std::sin(E)-mean)/(1-.4*std::cos(E));E-=d;if(std::fabs(d)<1e-15)break;}
                    V3 exact={std::cos(E)-.4,std::sqrt(.84)*std::sin(E),0};
                    phase_error=norm(minus(minus(val(q.x,1),val(q.x,0)),exact));
                    good=good&&phase_error<=1e-6&&std::fabs(q.physical_energy_change)<=1e-7;
                }
                pass=pass&&good;
                std::printf("RESULT case=%d k=%.0f rep=%d seconds=%.9g a_error=%.9g e_error=%.9g h_error=%.9g outer_error=%.9g Hsd_error=%.9g physical_dE=%.9g L_error=%.9g slowed_phase_error=%.9g steps=%ld rejected=%ld pass=%d\n",kind,k,rep,q.seconds,da,de,dh,dx,q.energy_error,q.physical_energy_change,q.L_error,phase_error,q.accepted,q.rejected,good);
            }
            double speed=ref.completed&&completed?median(base_times)/median(times):0;std::printf("TIMING case=%d k=%.0f median_speedup=%.6g accuracy_pass=%d speed_bar_pass=%d\n",kind,k,speed,pass,speed>=1.1);
            if(!is_encounter(kind)&&!(speed>=1.1))speed_failures++;
            if(!pass){if(is_encounter(kind))hard_failures++;else failures++;}
        }
    }
    for(int feature=1;feature<=6;feature++){
        State ref=run(0,1,false,feature),q=run(0,16,true,feature);bool pass=ref.completed&&q.completed&&q.applied==1&&bits(ref,q);
        std::printf("FALLBACK feature=%d applied=%.1f bitwise=%d pass=%d\n",feature,q.applied,bits(ref,q),pass);if(!pass)failures++;
    }
    std::printf("SUMMARY correctness_failures=%d strong_encounter_failures=%d speed_failures=%d phase1_pass=%d production_acceptable=0\n",failures,hard_failures,speed_failures,failures==0&&hard_failures==0&&speed_failures==0);
    return failures||hard_failures||speed_failures?1:0;
}
