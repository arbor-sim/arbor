#pragma once

namespace nestmc {

class i_clamp {
    public:

    using value_type = double;

    i_clamp(value_type del, value_type dur, value_type amp)
    :   delay_(del),
        duration_(dur),
        amplitude_(amp)
    {}

    value_type delay() const {
        return delay_;
    }
    value_type duration() const {
        return duration_;
    }
    value_type amplitude() const {
        return amplitude_;
    }

    void set_delay(value_type d) {
        delay_ = d;
    }
    void set_duration(value_type d) {
        duration_ = d;
    }
    void set_amplitude(value_type a) {
        amplitude_ = a;
    }

    // current is set to amplitude for time in the half open interval:
    //      t \in [delay, delay+duration)
    value_type amplitude(double t) {
        if(t>=delay_ && t<(delay_+duration_)) {
            return amplitude_;
        }
        return 0;
    }

    private:

    value_type delay_     = 0;
    value_type duration_  = 0;
    value_type amplitude_ = 0;
};

} // namespace nestmc
