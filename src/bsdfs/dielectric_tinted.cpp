#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/ior.h>
#include <stdlib.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class DielectricTinted final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    DielectricTinted(const Properties &props) : Base(props) {

        // // Specifies the internal index of refraction at the interface
        // ScalarFloat int_ior = lookup_ior(props, "int_ior", "bk7");

        // // Specifies the external index of refraction at the interface
        // ScalarFloat ext_ior = lookup_ior(props, "ext_ior", "air");

        // if (int_ior < 0 || ext_ior < 0)
        //     Throw("The interior and exterior indices of refraction must"
        //           " be positive!");

        // m_eta = int_ior / ext_ior;

        m_eta = 1.33;

        if(props.has_property("eta"))
            m_eta = props.get<ScalarFloat>("eta");
        
        m_tint = props.get<ScalarPoint3f>("tint");

        // if (props.has_property("specular_reflectance"))
        //     m_specular_reflectance   = props.texture<Texture>("specular_reflectance", 1.f);
        // if (props.has_property("specular_transmittance"))
        //     m_specular_transmittance = props.texture<Texture>("specular_transmittance", 1.f);

        m_components.push_back(BSDFFlags::DeltaReflection | BSDFFlags::FrontSide |
                               BSDFFlags::BackSide);
        m_components.push_back(BSDFFlags::DeltaTransmission | BSDFFlags::FrontSide |
                               BSDFFlags::BackSide/* | BSDFFlags::NonSymmetric*/);

        m_flags = m_components[0] | m_components[1];
        dr::set_attr(this, "flags", m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("tint", m_tint, +ParamFlags::Differentiable);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f & /* sample2 */,
                                             Mask active) const override {
        // ??? Supposed to mask part of computation using active?
        // MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        bool has_reflection   = ctx.is_enabled(BSDFFlags::DeltaReflection, 0),
             has_transmission = ctx.is_enabled(BSDFFlags::DeltaTransmission, 1);

        // Evaluate the Fresnel equations for unpolarized illumination
        Float cos_theta_i = Frame3f::cos_theta(si.wi);

        auto [r_i, cos_theta_t, eta_it, eta_ti] = fresnel(cos_theta_i, Float(m_eta));
        Float t_i = dr::maximum(1.f - r_i, 0.f);

        // Lobe selection
        BSDFSample3f bs = BSDFSample3f();// dr::zeros<BSDFSample3f>();
        Mask selected_r;
        selected_r = sample1 <= r_i && active;

        // if (likely(has_reflection && has_transmission)) {
        //     selected_r = sample1 <= r_i && active;
        //     bs.pdf = dr::detach(dr::select(selected_r, r_i, t_i));
        // } else {
        //     if (has_reflection || has_transmission) {
        //         selected_r = Mask(has_reflection) && active;
        //         bs.pdf = 1.f;
        //     } else {
        //         return { bs, 0.f };
        //     }
        // }
        // Mask selected_t = !selected_r && active;

        bs.pdf = dr::select(selected_r, r_i, t_i);
        bs.sampled_component = dr::select(selected_r, UInt32(0), UInt32(1));
        bs.sampled_type      = dr::select(selected_r, UInt32(+BSDFFlags::DeltaReflection),
                                                      UInt32(+BSDFFlags::DeltaTransmission));

        bs.wo = dr::select(selected_r,
                           reflect(si.wi),
                           refract(si.wi, cos_theta_t, eta_ti));

        bs.eta = dr::select(selected_r, Float(1.f), eta_it);

        // For reflection, tint based on the incident angle (more tint at grazing angle)
        auto value_r = dr::lerp(Color3f(m_tint), Color3f(1.f), dr::clamp(cos_theta_i, 0.f, 1.f));

        auto value_t = Color3f(1.f) * dr::sqr(eta_ti);

        auto value = dr::select(selected_r, value_r, value_t);

        return { bs, value };
    }

    Spectrum eval(const BSDFContext & /* ctx */, const SurfaceInteraction3f & /* si */,
                  const Vector3f & /* wo */, Mask /* active */) const override {
        return 0.f;
    }

    Float pdf(const BSDFContext & /* ctx */, const SurfaceInteraction3f & /* si */,
              const Vector3f & /* wo */, Mask /* active */) const override {
        return 0.f;
    }

    std::pair<Spectrum, Float> eval_pdf (const BSDFContext & /*ctx*/, const SurfaceInteraction3f & /*si*/,
              const Vector3f & /*wo*/, Mask /*active*/) const override {
        return std::pair<Spectrum, Float> (0.f, 0.f);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DielectricTinted[" << std::endl;
        oss << "  eta = " << m_eta << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ScalarFloat m_eta;
    ScalarPoint3f m_tint;
};

MI_IMPLEMENT_CLASS_VARIANT(DielectricTinted, BSDF)
MI_EXPORT_PLUGIN(DielectricTinted, "Smooth dielectric")
NAMESPACE_END(mitsuba)
