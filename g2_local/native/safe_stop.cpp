// Narrow extension: reuse the Python-owned Robot; never construct another one.
#include <pybind11/pybind11.h>
#include <gdk/robot.h>
namespace py = pybind11;
using agibot::gdk::Robot;
using Mode = agibot::gdk::MotionControlModeFull;

static Mode request(int control_mode) {
    if (control_mode != 1 && control_mode != 3)
        throw std::invalid_argument("Expected control mode 1 or 3");
    Mode mode;
    mode.input_source = Mode::InputSource::INPUT_GDK;
    mode.target = Mode::Target::TARGET_LEFT_ARM;
    mode.control_mode = static_cast<Mode::ControlMode>(control_mode);
    mode.safe_mode = Mode::SafeMode::SAFE_STOP;
    // Keep the SDK default priority. No priority escalation or reset API.
    return mode;
}

PYBIND11_MODULE(_gdk_safe_stop, m) {
    m.def("robot_type_registered", []() {
        return py::detail::get_type_info(typeid(Robot), false) != nullptr;
    });
    m.def("check_robot", [](Robot &) { return true; }); // cast only, no SDK call
    m.def("request_fields", [](int control_mode) {
        auto mode = request(control_mode);
        return py::make_tuple(int(mode.input_source), int(mode.target),
                              int(mode.control_mode), int(mode.safe_mode), mode.priority);
    });
    m.def("request_left_safe_stop", [](Robot &robot, int control_mode) {
        auto mode = request(control_mode);
        py::gil_scoped_release release;
        auto result = robot.SetControlModeFull(mode);
        if (result != agibot::gdk::GDKRes::kSuccess)
            throw std::runtime_error("GDK left SAFE_STOP request failed: " +
                                     std::to_string(int(result)));
        return 0; // request acknowledgement only, NOT physical stop evidence
    });
}
