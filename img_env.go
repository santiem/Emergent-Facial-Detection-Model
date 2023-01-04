// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image"
	"log"
	"math/rand"

	"strings"

	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
)

// ImgEnv presents images from a list of image files, using V1 simple and complex filtering.
// images are just selected at random each trial -- nothing fancy here.
type ImgEnv struct {
	Nm         string        `desc:"name of this environment"`
	Dsc        string        `desc:"description of this environment"`
	ImageFiles []string      `desc:"paths to images"`
	Images     []*image.RGBA `desc:"images (preload for speed)"`
	ImageIdx   env.CurPrvInt `desc:"current image index"`
	Vis        Vis           `desc:"visual processing params"`
	Run        env.Ctr       `view:"inline" desc:"current run of model as provided during Init"`
	Epoch      env.Ctr       `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial      env.Ctr       `view:"inline" desc:"trial is the step counter within epoch"`
	Output     etensor.Float32

	//OrigImg    etensor.Float32 `desc:"original image prior to random transforms"`
}

func (ev *ImgEnv) Name() string { return ev.Nm }
func (ev *ImgEnv) Desc() string { return ev.Dsc }

func (ev *ImgEnv) Validate() error {
	return nil
}

func (ev *ImgEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (ev *ImgEnv) States() env.Elements {
	isz := ev.Vis.ImgSize
	sz := ev.Vis.V1AllTsr.Shapes()
	nms := ev.Vis.V1AllTsr.DimNames()
	els := env.Elements{
		{"Image", []int{isz.Y, isz.X}, []string{"Y", "X"}},
		{"V1", sz, nms},
		{"Output", []int{1, 1}, []string{"Y", "X"}}}
	return els
}

func (ev *ImgEnv) State(element string) etensor.Tensor {
	switch element {
	case "Image":
		return &ev.Vis.ImgTsr
	case "V1":
		return &ev.Vis.V1AllTsr
	case "Output":
		return &ev.Output
	}
	return nil
}

func (ev *ImgEnv) Actions() env.Elements {
	return nil
}

func (ev *ImgEnv) Defaults() {
	ev.Vis.Defaults()
}

func (ev *ImgEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Output.SetShape([]int{1, 1}, nil, []string{"Y", "X"}) //added
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

func (ev *ImgEnv) Step() bool {
	ev.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if ev.Trial.Incr() { // if true, hit max, reset to 0
		ev.Epoch.Incr()
	}
	ev.PickRndImage()
	ev.FilterImg()
	// debug only:
	// img := ev.Images[ev.ImageIdx.Cur]
	// vfilter.RGBToGrey(img, &ev.OrigImg, 0, false) // pad for filt, bot zero
	return true
}

// DoImage processes specified image number
func (ev *ImgEnv) DoImage(imgNo int) {
	ev.ImageIdx.Set(imgNo)
	ev.FilterImg()
}

func (ev *ImgEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *ImgEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*ImgEnv)(nil)

// SetOutput sets the output
func (ev *ImgEnv) SetOutput(out int) {
	ev.Output.SetZeros()
	//ev.Output.SetFloat1D(out, 1)
	for i := 0; i < len(ev.Images); i++ {
		str := ev.ImageFiles[ev.ImageIdx.Cur]
		substr := "s"
		if strings.Contains(str, substr) {
			ev.Output.SetFloat1D(out, 1)
		} else {
			ev.Output.SetFloat1D(out, 0)
		}
	}
}

// PickRndImage picks an image at random
func (ev *ImgEnv) PickRndImage() {
	nimg := len(ev.Images)
	ev.ImageIdx.Set(rand.Intn(nimg))
	ev.SetOutput(0)
}

// FilterImg filters the image using new random xforms
func (ev *ImgEnv) FilterImg() {
	img := ev.Images[ev.ImageIdx.Cur]
	// following logic first extracts a sub-image of 2x the ultimate filtered size of image
	// from original image, which greatly speeds up the xform processes, relative to working
	// on entire 800x600 original image

	//insz := ev.Vis.Geom.In.Mul(2) // target size * 2
	//ibd := oimg.Bounds()
	//isz := ibd.Size()
	//irng := isz.Sub(insz)
	//var st image.Point
	//st.X = rand.Intn(irng.X)
	//st.Y = rand.Intn(irng.Y)
	//ed := st.Add(insz)
	//simg := oimg.SubImage(image.Rectangle{Min: st, Max: ed})
	ev.Vis.Filter(img)
}

// OpenImages opens all the images
func (ev *ImgEnv) OpenImages() error {
	nimg := len(ev.ImageFiles)
	if len(ev.Images) != nimg {
		ev.Images = make([]*image.RGBA, nimg)
	}
	var lsterr error
	for i, fn := range ev.ImageFiles {
		img, err := gi.OpenImage(fn)
		if err != nil {
			log.Println(err)
			lsterr = err
			continue
		}
		ev.Images[i] = gi.ImageToRGBA(img)
	}
	return lsterr
}
