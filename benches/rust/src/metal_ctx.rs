//! Metal device context: device, queue, pipeline compilation, buffer management.

use metal::*;
use std::collections::HashMap;

pub struct MetalCtx {
    pub device: Device,
    pub queue: CommandQueue,
    pipelines: HashMap<String, ComputePipelineState>,
}

impl MetalCtx {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        Some(Self {
            device,
            queue,
            pipelines: HashMap::new(),
        })
    }

    /// Compile a Metal shader function from source. Caches pipelines by name.
    pub fn pipeline(&mut self, name: &str, source: &str) -> &ComputePipelineState {
        if !self.pipelines.contains_key(name) {
            let opts = CompileOptions::new();
            opts.set_fast_math_enabled(true);
            let lib = self
                .device
                .new_library_with_source(source, &opts)
                .unwrap_or_else(|e| panic!("Metal compile error for '{name}': {e}"));
            let func = lib
                .get_function(name, None)
                .unwrap_or_else(|e| panic!("Metal function '{name}' not found: {e}"));
            let pso = self
                .device
                .new_compute_pipeline_state_with_function(&func)
                .unwrap_or_else(|e| panic!("Metal pipeline error for '{name}': {e}"));
            self.pipelines.insert(name.to_string(), pso);
        }
        &self.pipelines[name]
    }

    /// Create a Metal buffer from a slice (copies data into Metal-managed memory).
    pub fn buffer_from_slice<T>(&self, data: &[T]) -> Buffer {
        let bytes = std::mem::size_of_val(data);
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create an empty Metal buffer.
    pub fn buffer_empty(&self, bytes: usize) -> Buffer {
        self.device.new_buffer(
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Dispatch a compute kernel: 1D grid.
    pub fn dispatch_1d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        n: usize,
    ) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), 0);
        }

        let tg_size = pipeline.thread_execution_width() as usize;
        let tg_count = (n + tg_size - 1) / tg_size;
        enc.dispatch_thread_groups(
            MTLSize::new(tg_count as u64, 1, 1),
            MTLSize::new(tg_size as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch with threadgroup memory (for reductions).
    pub fn dispatch_reduce(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        n: usize,
        threadgroup_mem_bytes: usize,
    ) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), 0);
        }
        enc.set_threadgroup_memory_length(0, threadgroup_mem_bytes as u64);

        let tg_size = pipeline.thread_execution_width() as usize;
        let tg_count = (n + tg_size - 1) / tg_size;
        enc.dispatch_thread_groups(
            MTLSize::new(tg_count as u64, 1, 1),
            MTLSize::new(tg_size as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch 2D grid (for matrix/stencil ops).
    pub fn dispatch_2d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        width: usize,
        height: usize,
    ) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), 0);
        }

        let tg_w = 16usize;
        let tg_h = 16usize;
        let grid_w = (width + tg_w - 1) / tg_w;
        let grid_h = (height + tg_h - 1) / tg_h;
        enc.dispatch_thread_groups(
            MTLSize::new(grid_w as u64, grid_h as u64, 1),
            MTLSize::new(tg_w as u64, tg_h as u64, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}
