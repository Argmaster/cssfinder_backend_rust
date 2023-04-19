// Copyright 2023 Krzysztof Wisniewski <argmaster.world@gmail.com>
//
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the “Software”), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be included in all copies
// or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
// CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use pyo3::{FromPyObject, PyAny, PyResult};

#[derive(Clone)]
pub enum AlgoMode {
    FSnQd,
    SBiPa,
    G3PaE3qD,
    G4PaE3qD,
}

impl FromPyObject<'_> for AlgoMode {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let variant_name = ob.getattr("name")?.extract::<String>()?;

        match variant_name.as_str() {
            "FSnQd" => Ok(AlgoMode::FSnQd),
            "SBiPa" => Ok(AlgoMode::SBiPa),
            "G3PaE3qD" => Ok(AlgoMode::G3PaE3qD),
            "G4PaE3qD" => Ok(AlgoMode::G4PaE3qD),
            _ => panic!("Unknown value."),
        }
    }
}
