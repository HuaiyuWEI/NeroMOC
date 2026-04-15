function masconData = averageFieldToMascons(fieldData, lonGrid, latGrid, ...
    lonBound1, lonBound2, latBound1, latBound2, flagAcross180)
% Average gridded data onto mascon boxes while preserving trailing dimensions.

numMascons = numel(lonBound1);
fieldSize = size(fieldData);
trailingSize = fieldSize(3:end);
numSlices = prod(trailingSize);
fieldData2D = reshape(fieldData, fieldSize(1), fieldSize(2), numSlices);
fieldDataFlat = reshape(fieldData2D, [], numSlices);
masconData2D = zeros(numMascons, numSlices);

for i = 1:numMascons
    if ~flagAcross180(i)
        ind = lonGrid >= lonBound1(i) & lonGrid < lonBound2(i) ...
            & latGrid >= latBound1(i) & latGrid < latBound2(i);
    else
        ind = (lonGrid >= lonBound1(i) | lonGrid < lonBound2(i)) ...
            & latGrid >= latBound1(i) & latGrid < latBound2(i);
    end
    values = fieldDataFlat(ind(:), :);
    masconData2D(i, :) = mean(values, 1, 'omitnan');
end

masconData = reshape(masconData2D, [numMascons, trailingSize]);
end
