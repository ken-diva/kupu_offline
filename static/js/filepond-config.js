// register plugin
FilePond.registerPlugin(FilePondPluginImagePreview);

// Get a reference to the file input element
const img_front = document.querySelector('input[id="img_front"]');
const img_back = document.querySelector('input[id="img_back"]');

// Create a FilePond instance
FilePond.create(img_front, {
	storeAsFile: true,
});
FilePond.create(img_back, {
	storeAsFile: true,
});
