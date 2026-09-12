#include "traceloom/analysis/clock_calibration.h"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
 int rank=std::stoi(argv[1]); std::vector<traceloom::ClockCalibrationObservation> obs; traceloom::ClockCalibrationObservation x;
 while(std::cin>>x.marker_id>>x.source_timestamp_ns>>x.target_timestamp_ns>>x.target_uncertainty_ns)obs.push_back(x);
 traceloom::ClockCalibrationOptions o; o.source_clock_domain="rank-"+std::to_string(rank)+"-device";o.target_clock_domain="rank-0-device";o.marker_contract="matched-provider-collective-name-type-count-candidate-v1";
 auto m=traceloom::fit_affine_clock_model(obs,o);if(m.status==traceloom::ClockCalibrationStatus::kInvalid){std::cerr<<m.reason;return 1;}
 std::cout<<std::setprecision(24)<<"{\"format\":\"traceloom.distributed-clock-model/v1\",\"rank\":"<<rank<<",\"reference_rank\":0,\"metric\":\"end\",\"status\":\"candidate_only\",\"source_clock_domain\":\""<<o.source_clock_domain<<"\",\"target_clock_domain\":\"rank-0-device\",\"marker_contract\":\""<<o.marker_contract<<"\",\"scale\":"<<m.scale<<",\"reference_source_ns\":"<<m.reference_source_ns<<",\"reference_target_ns\":"<<m.reference_target_ns<<",\"drift_ppm\":"<<m.drift_ppm<<",\"offset_ns\":"<<m.offset_ns<<",\"holdout_p95_ns\":"<<m.absolute_residual_p95_ns<<",\"holdout_max_ns\":"<<m.absolute_residual_max_ns<<",\"observations\":"<<m.input_observation_count<<",\"holdout_count\":"<<m.validation_observation_count<<"}\n";
}
