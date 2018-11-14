import sys
import unittest
import itertools

import numpy as np

import lsst.utils.tests
import lsst.afw.detection
import lsst.afw.geom
import lsst.afw.image

from lsst.meas.deblender import SourceDeblendTask


class ForcedDecorrelationTestCase(lsst.utils.tests.TestCase):
    @lsst.utils.tests.debugger(Exception)
    def testIt(self):
        psfSigma = 3.21
        spacing = 6.0
        bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(12345, 67890),
                                   lsst.afw.geom.Extent2I(345, 321))
        forcedDecorrelationRadius = 100

        psfWidth = 2*int(5*psfSigma) + 1
        psf = lsst.afw.detection.GaussianPsf(psfWidth, psfWidth, psfSigma)
        exposure = lsst.afw.image.makeExposure(lsst.afw.image.MaskedImageF(bbox))
        exposure.setPsf(psf)
        exposure.variance.set(1.0e-6)

        spans = lsst.afw.geom.SpanSet()
        peaks = []
        psfSpacing = psfSigma*spacing
        for xy in itertools.product(np.arange(psfSpacing, bbox.getWidth() - psfSpacing, psfSpacing),
                                    np.arange(psfSpacing, bbox.getHeight() - psfSpacing, psfSpacing)):
            center = lsst.afw.geom.Point2D(*xy) + lsst.afw.geom.Extent2D(bbox.getMin())
            pixel = lsst.afw.geom.Point2I(center + lsst.afw.geom.Extent2I(0.5, 0.5))
            psfImage = psf.computeImage(center)
            exposure.image[psfImage.getBBox(), lsst.afw.image.PARENT] += psfImage.convertF()
            spans = spans.union(lsst.afw.geom.SpanSet.fromShape(int(psfSpacing),
                                                                lsst.afw.geom.Stencil.CIRCLE, pixel))
            peaks.append(pixel)

        spans.setMask(exposure.mask, exposure.mask.getPlaneBitMask("DETECTED"))

        numPeaks = len(peaks)

        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        config = SourceDeblendTask.ConfigClass()
        config.forcedDecorrelationRadius = forcedDecorrelationRadius
        config.maxFootprintArea = 4*forcedDecorrelationRadius**2

        task = SourceDeblendTask(schema, config=config)

        def makeCatalog():
            catalog = lsst.afw.table.SourceCatalog(schema)
            catalog.reserve(numPeaks + 1)
            footprint = lsst.afw.detection.Footprint(spans)
            assert footprint.isContiguous(), "Footprint isn't contiguous"
            for pp in peaks:
                newPeak = footprint.getPeaks().addNew()
                newPeak.setIx(pp.getX())
                newPeak.setIy(pp.getY())
                newPeak.setFx(pp.getX())
                newPeak.setFy(pp.getY())

            source = catalog.addNew()
            source.setFootprint(footprint)
            return catalog

        catalog1 = makeCatalog()
        task.run(exposure, catalog1)

        def getPosition(source):
            return tuple([*ss.getFootprint().getPeaks()[0].getI()])

        self.assertEqual(len(catalog1), numPeaks + 1)
        self.assertTrue(catalog1[0].get("deblend_forcedDecorr"))
        self.assertFalse(catalog1[0].get("deblend_parentTooBig"))
        self.assertEqual(catalog1[0].get("parent"), 0)
        self.assertEqual(catalog1[0].get("id"), 1)
        self.assertEqual(len(set(catalog1["id"])), len(catalog1))
        sources = {}
        for ss in catalog1[1:]:
            self.assertTrue(ss.get("deblend_forcedDecorr"))
            self.assertFalse(ss.get("deblend_parentTooBig"))
            self.assertEqual(len(ss.getFootprint().getPeaks()), 1)
            self.assertTrue(ss.getFootprint().isHeavy())
            self.assertEqual(ss.get("parent"), 1)
            sources[getPosition(ss)] = ss

        catalog2 = makeCatalog()

        config = SourceDeblendTask.ConfigClass()
        config.forcedDecorrelationRadius = 0  # Disable forced decorrelation
        config.maxFootprintArea = bbox.getArea() + 1
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        task = SourceDeblendTask(schema, config=config)
        task.run(exposure, catalog2)

        self.assertEqual(len(catalog2), numPeaks + 1)
        self.assertFalse(catalog2[0].get("deblend_forcedDecorr"))
        self.assertFalse(catalog2[0].get("deblend_parentTooBig"))
        self.assertEqual(catalog2[0].get("id"), 1)
        self.assertEqual(catalog2[0].get("parent"), 0)
        self.assertEqual(len(set(catalog2["id"])), len(catalog2))
        self.assertFootprintsEqual(catalog1[0].getFootprint(), catalog2[0].getFootprint())
        for ss in catalog2[1:]:
            self.assertFalse(ss.get("deblend_forcedDecorr"))
            self.assertFalse(ss.get("deblend_parentTooBig"))
            self.assertEqual(len(ss.getFootprint().getPeaks()), 1)
            self.assertTrue(ss.getFootprint().isHeavy())
            self.assertEqual(ss.get("parent"), 1)

            other = sources[getPosition(ss)]
            self.assertFootprintsEqual(ss.getFootprint(), other.getFootprint())

    def assertFootprintsEqual(self, lhs, rhs):
        self.assertEqual(lhs.getBBox(), rhs.getBBox())
        self.assertEqual(lhs.getSpans(), rhs.getSpans())
        self.assertEqual(lhs.isHeavy(), rhs.isHeavy())
        self.assertEqual(lhs.getPeaks().getSchema(), rhs.getPeaks().getSchema())
        self.assertEqual(set([tuple([*pp.getI()]) for pp in lhs.getPeaks()]),
                         set([tuple([*pp.getI()]) for pp in rhs.getPeaks()]))
        if lhs.isHeavy():
            np.testing.assert_almost_equal(lhs.getImageArray(), rhs.getImageArray())
            np.testing.assert_equal(lhs.getMaskArray(), rhs.getMaskArray())
            np.testing.assert_almost_equal(lhs.getVarianceArray(), rhs.getVarianceArray())


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules[__name__])
    unittest.main(failfast=True)
