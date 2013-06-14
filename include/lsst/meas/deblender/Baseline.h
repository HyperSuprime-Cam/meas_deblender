// -*- LSST-C++ -*-
#if !defined(LSST_DEBLENDER_BASELINE_H)
#define LSST_DEBLENDER_BASELINE_H
//!

#include <vector>
#include <utility>

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/Peak.h"

namespace lsst {
    namespace meas {
        namespace deblender {

            template <typename ImagePixelT,
                      typename MaskPixelT=lsst::afw::image::MaskPixel,
                      typename VariancePixelT=lsst::afw::image::VariancePixel>
            class BaselineUtils {

            public:
                typedef typename lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> MaskedImageT;
                typedef typename lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::Ptr MaskedImagePtrT;
                typedef typename lsst::afw::image::Image<ImagePixelT> ImageT;
                typedef typename lsst::afw::image::Image<ImagePixelT>::Ptr ImagePtrT;
                typedef typename lsst::afw::image::Mask<MaskPixelT> MaskT;
                typedef typename lsst::afw::image::Mask<MaskPixelT>::Ptr MaskPtrT;

                typedef typename lsst::afw::detection::Footprint FootprintT;
                typedef typename lsst::afw::detection::Footprint::Ptr FootprintPtrT;
                typedef typename lsst::afw::detection::HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT> HeavyFootprintT;

                typedef typename boost::shared_ptr<lsst::afw::detection::HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT> > HeavyFootprintPtr;

                static std::vector<double>
                fitEllipse(ImageT const& img,
                           double bkgd, double xc, double yc);

                static
                lsst::afw::detection::Footprint::Ptr
                symmetrizeFootprint(lsst::afw::detection::Footprint const& foot,
                                    int cx, int cy);

                static
                std::pair<MaskedImagePtrT, FootprintPtrT>
                buildSymmetricTemplate(MaskedImageT const& img,
                                       lsst::afw::detection::Footprint const& foot,
                                       lsst::afw::detection::Peak const& pk,
                                       double sigma1,
                                       bool minZero);

                static void
                medianFilter(MaskedImageT const& img,
                             MaskedImageT & outimg,
                             int halfsize);

                static void
                makeMonotonic(MaskedImageT & img,
                              lsst::afw::detection::Footprint const& foot,
                              lsst::afw::detection::Peak const& pk,
                              double sigma1);


                static const int ASSIGN_STRAYFLUX                          = 0x1;
                static const int STRAYFLUX_TO_POINT_SOURCES_WHEN_NECESSARY = 0x2;
                static const int STRAYFLUX_TO_POINT_SOURCES_ALWAYS         = 0x4;
                // split flux according to the closest distance to the template?
                // (default is according to distance to the peak)
                static const int STRAYFLUX_R_TO_FOOTPRINT                  = 0x8;
                // swig doesn't seem to understand std::vector<MaskedImagePtrT>...
                static
                std::vector<typename lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::Ptr>
                apportionFlux(MaskedImageT const& img,
                              lsst::afw::detection::Footprint const& foot,
                              std::vector<typename lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::Ptr> templates,
                              std::vector<boost::shared_ptr<lsst::afw::detection::Footprint> > templ_footprints,
                              //
                              ImagePtrT templ_sum,
                              std::vector<bool> const& ispsf,
                              std::vector<int>  const& pkx,
                              std::vector<int>  const& pky,
                              std::vector<boost::shared_ptr<typename lsst::afw::detection::HeavyFootprint<ImagePixelT,MaskPixelT,VariancePixelT> > > & strays,
                              int strayFluxOptions
                     );

                static
                std::vector<boost::shared_ptr<typename lsst::afw::detection::HeavyFootprint<ImagePixelT,MaskPixelT,VariancePixelT> > >
                getEmptyStrayFluxList() {
                    return std::vector<boost::shared_ptr<typename lsst::afw::detection::HeavyFootprint<ImagePixelT,MaskPixelT,VariancePixelT> > >();
                };

                /*** This should move to HeavyFootprint.cc ***/
                static
                boost::shared_ptr<lsst::afw::detection::HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT> >
                mergeHeavyFootprints(
                    lsst::afw::detection::HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT> const& h1,
                    lsst::afw::detection::HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT> const& h2);

            };
        }
    }
}

#endif
