function [ outputImage, newHeight, newWidth ] = resize_image( inputImage, vertPatchSize, horPatchSize, addPadding )
%RESIZE_IMAGE Resize image so that it is divisible by the patch size.
%
%	inputImage:		input image
%	vertPatchSize:	vertical patch size
%	horPatchSize:	horizontal patch size
%	addPadding:		1 = add padding; 0 = crop image (default)
%	outputImage:	output image
%	newHeight:		new height
%	newWidth:		new width

[height, width] = size(inputImage);

if nargin < 4 || ~addPadding

	newHeight = height - mod(height, vertPatchSize);
	newWidth = width - mod(width, horPatchSize);

	outputImage = inputImage(1:newHeight, 1:newWidth);

else

	% Determine extra height to pad
	if mod(height, vertPatchSize) == 0
		extra_rows = 0;
	else
		extra_rows = vertPatchSize - mod(height, vertPatchSize);
	end

	% Determine extra width to pad
	if mod(width, horPatchSize) == 0
		extra_cols = 0;
	else
		extra_cols = horPatchSize - mod(width, horPatchSize);
	end

	% Pad image to new height and width
	outputImage = padarray(inputImage, [extra_rows extra_cols], 'symmetric', 'post');

end

[newHeight, newWidth] = size(outputImage);

end